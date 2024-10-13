# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import sys
sys.path.append("/mnt/petrelfs/shaojie/code/DiT/")

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import numpy as np
import torch.nn.functional as F

import torch.distributed as dist
import shutil
from glob import glob
from time import time
from PIL import Image
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
from tqdm.auto import tqdm
import random

from pqf.utils.model_size import compute_model_nbits
from pqf.compression.model_compression import compress_model
from pqf.utils.config_loader import load_config
from distributed import init_distributed_mode
from pq.low_rank_models import DiT_uv_models


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
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
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
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
    """
    End DDP training.
    """
    dist.destroy_process_group()

def main(args):
    # Setup DDP:
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '5678'
    # dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)

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
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
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
    diffusion = create_diffusion(timestep_respacing="")

    # pq model
    file_path = os.path.dirname(__file__)
    default_config = os.path.join(file_path, "../pqf/config/train_dit.yaml")
    if rank == 0:
        shutil.copy(default_config, experiment_dir)
    config = load_config(file_path, default_config_path=default_config)
    model_config = config["model"]
    compression_config = model_config["compression_parameters"]

    uncompressed_model_size_bits = compute_model_nbits(model)
    num_layers = 28
    fc1_len = [128] * num_layers
    fc2_len = [128] * num_layers
    qkv_len = [128] * num_layers
    proj_len = [128] * num_layers
    adaln_len = [128] * num_layers

    fc1_use_uv = [True] * num_layers
    fc2_use_uv = [True] * num_layers
    qkv_use_uv = [True] * num_layers
    proj_use_uv = [True] * num_layers
    adaln_use_uv = [True] * num_layers

    model_uv = DiT_uv_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        qkv_use_uv=qkv_use_uv, proj_use_uv=proj_use_uv, fc1_use_uv=fc1_use_uv, fc2_use_uv=fc2_use_uv, adaln_use_uv=adaln_use_uv,
        qkv_len=qkv_len, proj_len=proj_len, fc1_len=fc1_len, fc2_len=fc2_len, adaln_len=adaln_len
    ).to(device)
    compressed_model_size_bits = compute_model_nbits(model_uv)
    logger.info(f"Uncompressed model size: {uncompressed_model_size_bits} bits")
    logger.info(f"Compressed model size: {compressed_model_size_bits} bits")
    logger.info(f"Compression ratio: {uncompressed_model_size_bits / compressed_model_size_bits:.2f}")

    # DDP
    model_uv = DDP(model_uv, device_ids=[rank])
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    del model

    # optimizer
    opt = torch.optim.AdamW(model_uv.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
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
    model_uv.train()  # important! This enables embedding dropout for classifier-free guidance
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            # loss_dict = diffusion.training_losses_distill(model_pq, model, x, t, model_kwargs)
            loss_dict = diffusion.training_losses(model_uv, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model_uv.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model_uv.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    parser.add_argument("--data-path", type=str, default="/mnt/petrelfs/share/images/train")
    parser.add_argument("--results-dir", type=str, default="results/low_rank")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)

    args = parser.parse_args()
    main(args)


