import numpy as np
import copy
import os
import random
from tqdm import tqdm
from glob import glob
from copy import deepcopy
import json
import torch
import argparse
import torch.nn as nn
from timm.layers.helpers import to_2tuple
import torch.distributed as dist

import shutil
import sys
sys.path.append("/mnt/petrelfs/shaojie/code/DiT/")
from models import DiT_models
from diffusion import create_diffusion
from pq.main_dit_calkl_s2_1 import reset_param, parse_option, merge_model
from download import find_model
from pq.low_rank_models import DiT_uv_models
from diffusers.models import AutoencoderKL
from pq.sample_pq import dit_generator
from distributed import init_distributed_mode
from pq.main_dit_calkl_s2 import create_logger, center_crop_arr, cleanup
from pq.train_lowrank import train_model
from pq.sample_ddp import sample
from pq.cal_optim_dit import get_blocks
from pqf.utils.model_size import compute_model_nbits
from pqf.utils.config_loader import load_config
from pqf.compression.model_compression import compress_model

def main(args):
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    # experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    experiment_dir = f"{args.results_dir}/011-DiT-XL-2"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    
    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = dit_generator('250', latent_size=latent_size, device=device)
    diffusion_dit = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    load_path = "results/low_rank/002-DiT-XL-2/dit_t_in_"
    uncompressed_model_size_bits = compute_model_nbits(model)
    # del model

    # DiT-XL/2-256
    percent = 0.9
    fc_space = range(1, 6)
    fc_len, fc_use_uv = {}, {}

    # Populate lengths and use_uv flags for each block
    for fc_idx in range(1, 6):
        if fc_idx in fc_space:
            fc_len[fc_idx], fc_use_uv[fc_idx] = get_blocks(percent, fc_idx, compress_all=True)
        else:
            fc_len[fc_idx] = [128] * 28
            fc_use_uv[fc_idx] = [False] * 28

    # Initialize model
    model_uv = DiT_uv_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        qkv_use_uv=fc_use_uv[3], proj_use_uv=fc_use_uv[4], 
        fc1_use_uv=fc_use_uv[1], fc2_use_uv=fc_use_uv[2], adaln_use_uv=fc_use_uv[5],
        qkv_len=fc_len[3], proj_len=fc_len[4], 
        fc1_len=fc_len[1], fc2_len=fc_len[2], adaln_len=fc_len[5]
    ).to(device)

    state_dict_merge = deepcopy(state_dict)
    # Iterate through blocks and merge model
    for fc_i in fc_space:
        for block_i in range(28):
            if fc_use_uv[fc_i][block_i]:
                state_dict_merge = merge_model(state_dict_merge, block_i, fc_i, fc_len[fc_i][block_i], load_path, logger)
                
    training = False
    msg = model_uv.load_state_dict(state_dict_merge)
    logger.info(msg)

    file_path = os.path.dirname(__file__)
    default_config = os.path.join(file_path, "../pqf/config/train_dit.yaml")
    if rank == 0:
        shutil.copy(default_config, experiment_dir)
    config = load_config(file_path, default_config_path=default_config)
    model_config = config["model"]
    compression_config = model_config["compression_parameters"]
    model_pq = compress_model(model_uv, **compression_config).cuda()
    compressed_model_size_bits = compute_model_nbits(model_pq)
    del model_uv
    
    logger.info(f"Uncompressed model size: {uncompressed_model_size_bits} bits")
    logger.info(f"Compressed model size: {compressed_model_size_bits} bits")
    logger.info(f"Compression ratio: {uncompressed_model_size_bits / compressed_model_size_bits:.2f}")

    training = False
    if not training:
        model_pq.eval()
        block_str = ''.join(map(str, fc_space))
        image_name = f"sample_allfc{block_str}_{percent:.1f}_pq".replace('.', '_')
        diffusion.forward_val(vae, model.forward, model_pq.forward, cfg=False, name=f"{experiment_dir}/{image_name}")
        # image_dir = f"{experiment_dir}/{image_name}"
        # sample(args, model_pq, vae, diffusion_dit, image_dir)
    else:
        train_model(args, logger, model_pq, vae, diffusion_dit, checkpoint_dir)


if __name__ == "__main__":
    args = parse_option()
    main(args)
    