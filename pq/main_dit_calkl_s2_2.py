import os
import time
import math
import random
import argparse
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from timm.utils import accuracy, AverageMeter
from diffusers.models import AutoencoderKL

import sys
sys.path.append("/mnt/petrelfs/shaojie/code/DiT/")
from glob import glob
from download import find_model
from diffusion import create_diffusion
from models import DiT_models
from pq.low_rank_models import DiT_uv_models

from distributed import init_distributed_mode
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from pq.sample_lowrank import dit_generator
from pq.main_dit_calkl_s2 import create_logger, center_crop_arr, cleanup
from pq.main_dit_calkl_s2_1 import reset_param, merge_model
from timm.utils import AverageMeter

import logging
from PIL import Image
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


class dit_evaluator(dit_generator):
    def __init__(self, timestep_respacing, latent_size, device):
        super(dit_evaluator, self).__init__(timestep_respacing, latent_size, device)

    def forward_val(self, model, model_uv, cfg=False):
        class_labels = list(range(16))
        bs = 8
        mse_loss_meter = AverageMeter()

        for i in range(0, len(class_labels), bs):
            label = class_labels[i:i + bs]
            z, model_kwargs = self.pre_process(label, cfg=cfg)
            img = z
            img_pq = z
            indices = list(range(self.num_timesteps))[::-1]
            indices_tqdm = tqdm(indices)

            for i in indices_tqdm:
                t = torch.tensor([i] * z.shape[0], device=self.device)
                with torch.no_grad():
                    map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
                    new_ts = map_tensor[t]
                    model_output = model(img, new_ts, **model_kwargs)
                    model_output_pq = model_uv(img_pq, new_ts, **model_kwargs)
                    img, img_pq = self.post_process(t, img, img_pq, model_output, model_output_pq)

            mse_loss = torch.mean((img - img_pq) ** 2)
            mse_loss_meter.update(mse_loss.item(), 1)

        return mse_loss_meter.avg

def parse_option():
    parser = argparse.ArgumentParser('DiT script', add_help=False)

    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    parser.add_argument("--data-path", type=str, default="./sampled_images")
    parser.add_argument("--results-dir", type=str, default="results/low_rank")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    return args

def main(args):
    init_distributed_mode(args)

    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/008-DiT-XL-2"  # Create an experiment folder
        # experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    
    latent_size = args.image_size // 8
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    load_path = "results/low_rank/002-DiT-XL-2/dit_t_in_"
    os.makedirs(os.path.join(experiment_dir, 'dim_loss'), exist_ok=True)
    criterion = dit_evaluator('250', latent_size, device)

    model_t = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    msg = model_t.load_state_dict(state_dict)
    model_t = DDP(model_t, device_ids=[rank])
    model_t.eval()

    num_layers = 28
    fc1_dim = [921.6] + np.arange(896, 32, -32).tolist()
    fc2_dim = [921.6] + np.arange(896, 32, -32).tolist()
    qkv_dim = [864] + np.arange(832, 32, -32).tolist()
    proj_dim = [576] + np.arange(544, 32, -32).tolist()
    adaln_dim = [987.4] + np.arange(960, 32, -32).tolist()

    for fc_i in range(1, 6):
        for block_i in range(28):
            if fc_i < 2 or (fc_i == 2 and block_i < 13):
                continue
            fc1_len, fc2_len, qkv_len, proj_len, adaln_len, fc1_use_uv, fc2_use_uv, qkv_use_uv, proj_use_uv, adaln_use_uv = reset_param(num_layers)
            if fc_i == 1:
                fc1_use_uv[block_i] = True
                dims = fc1_dim
            elif fc_i == 2:
                fc2_use_uv[block_i] = True
                dims = fc2_dim
            elif fc_i == 3:
                qkv_use_uv[block_i] = True
                dims = qkv_dim
            elif fc_i == 4:
                proj_use_uv[block_i] = True
                dims = proj_dim
            elif fc_i == 5:
                adaln_use_uv[block_i] = True
                dims = adaln_dim
            
            for dim_i in range(len(dims)):
                dim = dims[dim_i]
                if fc_i == 1:
                    fc1_len[block_i] = dim
                elif fc_i == 2:
                    fc2_len[block_i] = dim
                elif fc_i == 3:
                    qkv_len[block_i] = dim
                elif fc_i == 4:
                    proj_len[block_i] = dim
                elif fc_i == 5:
                    adaln_len[block_i] = dim
                
                if dim_i == 0:
                    model = DiT_models[args.model](
                        input_size=latent_size,
                        num_classes=args.num_classes
                    ).to(device)
                    msg = model.load_state_dict(state_dict)
                else:
                    model = DiT_uv_models[args.model](
                        input_size=latent_size,
                        num_classes=args.num_classes,
                        qkv_use_uv=qkv_use_uv, proj_use_uv=proj_use_uv, fc1_use_uv=fc1_use_uv, fc2_use_uv=fc2_use_uv, adaln_use_uv=adaln_use_uv,
                        qkv_len=qkv_len, proj_len=proj_len, fc1_len=fc1_len, fc2_len=fc2_len, adaln_len=adaln_len
                    ).to(device)
                    state_dict_merge = merge_model(deepcopy(state_dict), block_i, fc_i, dim, load_path, logger)
                    msg = model.load_state_dict(state_dict_merge)

                n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"number of params: {n_parameters}")
                logger.info(msg)
                model = DDP(model, device_ids=[rank])
                model.eval()

                loss = criterion.forward_val(model_t.forward, model.forward)
                file_path = f"{experiment_dir}/dim_loss/{block_i}_{fc_i}.txt"
                with open(file_path, 'a') as file:
                    file.write(f"{dim}, {loss}\n")

                if block_i == 27 and fc_i == 5:
                    logger.info("Done.")
                else:
                    del model
    dist.barrier()
    
if __name__ == '__main__':
    args = parse_option()
    main(args)