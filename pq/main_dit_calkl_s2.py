import os
import time
import math
import random
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

import logging
from PIL import Image
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

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


def plot_3d_surface(matrix, ax, title, vmin, vmax):
    """Efficient 3D surface plotting."""
    x = np.arange(matrix.shape[0])
    y = np.arange(matrix.shape[1])
    X, Y = np.meshgrid(x, y)
    Z = matrix.cpu().numpy()

    # Use faster plotting without edge color and unnecessary details
    surf = ax.plot_surface(X, Y, Z.T, cmap='viridis')

    # Setting labels and titles with reduced font sizes for efficiency
    ax.set_title(title, fontsize=10)
    # ax.set_zlim(vmin, vmax)
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)

def calculate_outliers(matrix, logger, name):
    matrix_np = matrix.cpu().numpy()
    mean = np.mean(matrix_np)
    std = np.std(matrix_np)
    
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    outliers = np.logical_or(matrix_np < lower_bound, matrix_np > upper_bound)
    outlier_ratio = np.sum(outliers) / matrix_np.size

    logger.info(f"{name} - Min: {matrix_np.min()}, Max: {matrix_np.max()}, Outliers: {outlier_ratio*100:.2f}%")
    return matrix_np.min(), matrix_np.max(), outlier_ratio

def merge_model(checkpoint, block_i, fc_i, dim_i, load_path, logger, plot_path):
    if fc_i == 1:
        file_name = "fc1"
        org_name = "fc1"
        w_u_name = "fc1_u"
        w_v_name = "fc1_v"
        layer_name = ".mlp."
    elif fc_i == 2:
        file_name = "fc2"
        org_name = "fc2"
        w_u_name = "fc2_u"
        w_v_name = "fc2_v"
        layer_name = ".mlp."
    elif fc_i == 3:
        file_name = "qkv"
        org_name = "qkv"
        w_u_name = "qkv_u"
        w_v_name = "qkv_v"
        layer_name = ".attn."
    elif fc_i == 4:
        file_name = "proj"
        org_name = "proj"
        w_u_name = "proj_u"
        w_v_name = "proj_v"
        layer_name = ".attn."
    elif fc_i == 5:
        file_name = "adaln"
        org_name = "1"
        w_u_name = "1"
        w_v_name = "2"
        layer_name = ".adaLN_modulation."

    u_name = load_path + file_name + "_" + str(block_i) + "_u.pt"
    v_name = load_path + file_name + "_" + str(block_i) + "_v.pt"
    avg_name = load_path + file_name + "_" + str(block_i) + "_avg.pt"

    u = torch.load(u_name, map_location='cpu').cpu()
    v = torch.load(v_name, map_location='cpu').cpu()
    avg = torch.load(avg_name, map_location='cpu').cpu()

    new_u = u[:, :dim_i]
    new_v = v[:dim_i, :]
    b = (torch.eye(new_u.shape[0]).cpu() - new_u @ new_v) @ avg

    original_w_name = "blocks." + str(block_i) + layer_name + org_name + ".weight"
    original_b_name = "blocks." + str(block_i) + layer_name + org_name + ".bias"
    new_w_name = "blocks." + str(block_i) + layer_name + w_u_name + ".weight"
    new_b_name = "blocks." + str(block_i) + layer_name + w_u_name + ".bias"
    original_w = checkpoint[original_w_name]
    original_b = checkpoint[original_b_name]

    new_w = new_v @ original_w
    new_b = original_b @ new_v.transpose(0, 1)

    del checkpoint[original_w_name]
    del checkpoint[original_b_name]
    checkpoint[new_w_name] = new_w
    checkpoint[new_b_name] = new_b

    k_split = new_w_name.split(".")
    k_split[-2] = w_v_name
    logger.info(".".join(k_split))
    checkpoint[".".join(k_split)] = new_u

    k_split[-1] = "bias"
    checkpoint[".".join(k_split)] = b.reshape(-1)

    min_w, max_w, outliers_w = calculate_outliers(original_w, logger, "Original W")
    min_new_w, max_new_w, outliers_new_w = calculate_outliers(new_w, logger, "New W")
    min_new_u, max_new_u, outliers_new_u = calculate_outliers(new_u, logger, "New U")

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})  # Add projection='3d' here
    global_min = min(min_w, min_new_w, min_new_u)
    global_max = max(max_w, max_new_w, max_new_u)

    plot_3d_surface(original_w, axs[0], 'Original W', vmin=global_min, vmax=global_max)
    plot_3d_surface(new_w, axs[1], 'New W', vmin=global_min, vmax=global_max)
    plot_3d_surface(new_u, axs[2], 'New U', vmin=global_min, vmax=global_max)

    plt.tight_layout()
    os.makedirs(os.path.join(plot_path, 'wdist_visual'), exist_ok=True)
    plot_path = os.path.join(plot_path, f'wdist_visual/{block_i}-{fc_i}-{dim_i}.jpg')
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

    return checkpoint


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
        shuffle=False,
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
    
    num_layers = 28
    fc1_len = [128] * num_layers
    fc2_len = [128] * num_layers
    qkv_len = [128] * num_layers
    proj_len = [128] * num_layers
    adaln_len = [128] * num_layers

    fc1_use_uv = [False] * num_layers
    fc2_use_uv = [False] * num_layers
    qkv_use_uv = [False] * num_layers
    proj_use_uv = [False] * num_layers
    adaln_use_uv = [False] * num_layers

    latent_size = args.image_size // 8
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    diffusion = create_diffusion(timestep_respacing="")
    load_path = "results/low_rank/002-DiT-XL-2/dit_t_in_"

    for block_i in range(28):
        for fc_i in range(1, 6):
            if fc_i == 1:
                fc1_use_uv[block_i] = True
            elif fc_i == 2:
                fc2_use_uv[block_i] = True
            elif fc_i == 3:
                qkv_use_uv[block_i] = True
            elif fc_i == 4:
                proj_use_uv[block_i] = True
            elif fc_i == 5:
                adaln_use_uv[block_i] = True

            model = DiT_uv_models[args.model](
                input_size=latent_size,
                num_classes=args.num_classes,
                qkv_use_uv=qkv_use_uv, proj_use_uv=proj_use_uv, fc1_use_uv=fc1_use_uv, fc2_use_uv=fc2_use_uv, adaln_use_uv=adaln_use_uv,
                qkv_len=qkv_len, proj_len=proj_len, fc1_len=fc1_len, fc2_len=fc2_len, adaln_len=adaln_len
            ).to(device)
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"number of params: {n_parameters}")
            state_dict = merge_model(state_dict, block_i, fc_i, 128, load_path, logger, plot_path=experiment_dir)
            msg = model.load_state_dict(state_dict)
            logger.info(msg)
            model = DDP(model, device_ids=[rank])

            validate(args, loader, model, vae, diffusion, device, logger, experiment_dir)
            if block_i == 27 and fc_i == 5:
                logger.info("Done.")
            else:
                del model
    
    checkpoint = {
        "model": model.module.state_dict(),
        "args": args
    }
    checkpoint_path = f"{experiment_dir}/ckpt.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    dist.barrier()


@torch.no_grad()
def validate(args, data_loader, model, vae, diffusion, device, logger, work_dir):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()

    for idx, (x, y) in enumerate(data_loader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        # t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        t = torch.linspace(0, diffusion.num_timesteps - 1, steps=x.shape[0], device=device).long()
        model_kwargs = dict(y=y)

        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()

        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = loss.item() / dist.get_world_size()

        loss_meter.update(loss, y.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(
        f'Test: [{idx}/{len(data_loader)}]\t'
        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        f'Mem {memory_used:.0f}MB')

if __name__ == '__main__':
    args = parse_option()
    main(args)