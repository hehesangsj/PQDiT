# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import math
import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers.models import AutoencoderKL

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

import sys
sys.path.append("/mnt/petrelfs/shaojie/code/DiT/")
import logging
from tqdm import tqdm
from PIL import Image
from glob import glob
try:
    from apex import amp
except ImportError:
    amp = None

from download import find_model
from diffusion import create_diffusion
from models import DiT_models
from distributed import init_distributed_mode

def create_logger(logging_dir):
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def cleanup():
    dist.destroy_process_group()

def parse_option():
    parser = argparse.ArgumentParser('DiT low rank compression script', add_help=False)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    parser.add_argument("--data-path", type=str, default="/mnt/petrelfs/share/images/train")
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
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    diffusion = create_diffusion(timestep_respacing="")

    model = DDP(model.to(device), device_ids=[rank])
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    loss = validate(args, loader, model, vae, diffusion, device, logger, experiment_dir)
    logger.info(f"Loss on dataset: {loss:.4f}")


def cal_cov(tensor_i, avg_tensor_i, ratio):
    u, s, vh = torch.linalg.svd(tensor_i, full_matrices=False)
    return  u, vh, avg_tensor_i

class output_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer. 
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None

def cal_output_sum_square(fc_output_list, fc_square_list, fc_sum_list, fc_count_list):
    for i in range(len(fc_output_list)):
        if len(fc_output_list[i].shape) == 3:
            tensor_i = fc_output_list[i].flatten(0, 1)
        elif len(fc_output_list[i].shape) == 2:
            tensor_i = fc_output_list[i]      
        fc_sum = torch.sum(tensor_i, dim=0)
        fc_square = tensor_i.T @ tensor_i
        fc_count_list[i] += tensor_i.shape[0]
        if fc_square_list[i] is None:
            fc_sum_list[i] = fc_sum
            fc_square_list[i] = fc_square
        else:
            fc_sum_list[i] += fc_sum
            fc_square_list[i] += fc_square

def cal_save_uvb(fc_square_list, fc_sum_list, fc_count_list, fc_ratio, save_name = "fc1", save_path= "test", logger=None):
    for i in range(len(fc_square_list)):
        logger.info(f"Save name: {save_name}, index: {i}")
        fc_square = fc_square_list[i]
        fc_sum = fc_sum_list[i]
        if fc_sum is not None:
            avg_tensor_i = fc_sum.reshape(-1, 1) / fc_count_list[i]
            cov_tensor_i = fc_square / fc_count_list[i] - avg_tensor_i @ avg_tensor_i.T
            u, v, b  = cal_cov(cov_tensor_i, avg_tensor_i, fc_ratio[i])
            u_name = save_path + save_name  + "_"  + str(i) + "_u.pt"
            torch.save(u, u_name)
            v_name = save_path + save_name  + "_" + str(i) + "_v.pt"
            torch.save(v, v_name)
            b_name = save_path + save_name  + "_" + str(i) + "_avg.pt"
            torch.save(b, b_name)

@torch.no_grad()
def validate(args, data_loader, model, vae, diffusion, device, logger, work_dir):
    model.eval()

    module_name = []
    hooks, hook_handles = [], []

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
            module_name.append(n)

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()

    num_layers = 28
    fc1_square_list, fc1_sum_list, fc1_count_list = [None] * num_layers,  [None] * num_layers, [0] * num_layers
    fc2_square_list, fc2_sum_list, fc2_count_list= [None] * num_layers, [None] * num_layers, [0] * num_layers
    qkv_square_list, qkv_sum_list, qkv_count_list= [None] * num_layers, [None] * num_layers, [0] * num_layers
    proj_square_list, proj_sum_list, proj_count_list= [None] * num_layers, [None] * num_layers, [0] * num_layers
    adaln_square_list, adaln_sum_list, adaln_count_list= [None] * num_layers, [None] * num_layers, [0] * num_layers
    fc1_ratio, fc2_ratio, qkv_ratio, proj_ratio, adaln_ratio =  [0] * num_layers, [0] * num_layers, [0] * num_layers, [0] * num_layers, [0] * num_layers
    
    iters = 10000

    for idx, (x, y) in enumerate(data_loader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        for hook in hooks:
            hook.clear()
        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        model_kwargs = dict(y=y)

        fc1_output_list = []
        fc2_output_list = []
        qkv_output_list = []
        proj_output_list = []
        adaln_output_list = []

        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        for cal in range(len(module_name)):
            if "blocks" in module_name[cal]:
                if "attn.qkv" in module_name[cal]:
                    qkv_output_list.append(hooks[cal].outputs)
                if "attn.proj" in module_name[cal]:
                    proj_output_list.append(hooks[cal].outputs)
                if "mlp.fc1" in module_name[cal]:
                    fc1_output_list.append(hooks[cal].outputs)
                if "mlp.fc2" in module_name[cal]:
                    fc2_output_list.append(hooks[cal].outputs)
                if "adaLN_modulation" in module_name[cal]:
                    adaln_output_list.append(hooks[cal].outputs)

        cal_output_sum_square(fc1_output_list, fc1_square_list, fc1_sum_list, fc1_count_list)
        cal_output_sum_square(fc2_output_list, fc2_square_list, fc2_sum_list, fc2_count_list)
        cal_output_sum_square(qkv_output_list, qkv_square_list, qkv_sum_list, qkv_count_list)
        cal_output_sum_square(proj_output_list, proj_square_list, proj_sum_list, proj_count_list)
        cal_output_sum_square(adaln_output_list, adaln_square_list, adaln_sum_list, adaln_count_list)

        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = loss.item() / dist.get_world_size()
        loss_meter.update(loss, y.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.log_every == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')
        
        if idx >= iters:
            break

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    save_path = work_dir + "/dit_t_in_"
    cal_save_uvb(adaln_square_list, adaln_sum_list, adaln_count_list, adaln_ratio, "adaln", save_path, logger )
    cal_save_uvb(fc1_square_list, fc1_sum_list, fc1_count_list, fc1_ratio, "fc1", save_path, logger)
    cal_save_uvb(fc2_square_list, fc2_sum_list, fc2_count_list, fc2_ratio, "fc2", save_path, logger )
    cal_save_uvb(qkv_square_list, qkv_sum_list, qkv_count_list, qkv_ratio, "qkv", save_path, logger )
    cal_save_uvb(proj_square_list, proj_sum_list, proj_count_list, proj_ratio, "proj", save_path, logger )

    return loss_meter.avg


if __name__ == '__main__':
    args = parse_option()
    main(args)
