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
import torchvision.transforms as transforms

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

import sys
sys.path.append("/mnt/petrelfs/shaojie/code/DiT/")
from glob import glob
from download import find_model
from diffusion import create_diffusion
from models import DiT_models
from distributed import init_distributed_mode
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import logging
from tqdm import tqdm
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
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--block-i', type=int, default=-1)
    parser.add_argument('--fc-i', type=int, default=-1)
    parser.add_argument('--dim-i', type=int, default=-1)

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

def merge_model(checkpoint, block_i, fc_i, dim_i, load_path):
    if fc_i == 1:
        fc_name = "fc1"
        layer_name = ".mlp."
    elif fc_i == 2:
        fc_name = "fc2"
        layer_name = ".mlp."
    elif fc_i == 3:
        fc_name = "qkv"
        layer_name = ".attn."
    elif fc_i == 4:
        fc_name = "proj"
        layer_name = ".attn."
    elif fc_i == 5:
        fc_name = "1"
        layer_name = ".adaLN_modulation."

    u_name = load_path +  fc_name + "_" + str(block_i) + "_u.pt"
    v_name = load_path + fc_name + "_" + str(block_i) + "_v.pt"
    avg_name = load_path + fc_name + "_" + str(block_i) + "_avg.pt"

    u = torch.load(u_name, map_location='cpu').cpu()
    v = torch.load(v_name, map_location='cpu').cpu()
    avg = torch.load(avg_name, map_location='cpu').cpu()

    new_u = u[:,:dim_i]
    new_v = v[:dim_i,:]
    b = (torch.eye(new_u.shape[0]).cpu()- new_u @ new_v) @ avg

    original_w_name = "blocks." + str(block_i) + layer_name + fc_name + ".weight"
    original_b_name = "blocks." + str(block_i) + layer_name + fc_name + ".bias"
    original_w = checkpoint[original_w_name]
    original_b = checkpoint[original_b_name]

    new_w = new_v @ original_w
    new_b = original_b @ new_v.transpose(0,1)

    checkpoint[original_w_name] = new_w
    checkpoint[original_b_name] = new_b

    k_split = original_w_name.split(".")
    k_split[-2] = "uv" + str((fc_i- 1)%2 + 1)
    print(".".join(k_split))
    checkpoint[".".join(k_split)] = new_u

    k_split[-1] = "bias"
    checkpoint[".".join(k_split)] = b.reshape(-1)

    for k, v in checkpoint.items():
        print(k, v.shape)
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
    
    fc1_len = [128] * 12
    fc2_len = [128] * 12
    qkv_len = [128] * 12
    proj_len = [128] * 12

    fc1_use_uv = [False] * 13
    fc2_use_uv = [False] * 13
    qkv_use_uv = [False] * 13
    proj_use_uv = [False] * 13

    if args.fc_i == 1:
        fc1_use_uv[args.block_i] = True
        fc1_len[args.block_i] = args.dim_i
    elif args.fc_i == 2:
        fc2_use_uv[args.block_i] = True
        fc2_len[args.block_i] = args.dim_i
    elif args.fc_i == 3:
        qkv_use_uv[args.block_i] = True
        qkv_len[args.block_i] = args.dim_i


    model = VisionTransformerAda(patch_size=16,  embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
            qkv_use_uv = qkv_use_uv,proj_use_uv=proj_use_uv,fc1_use_uv = fc1_use_uv,fc2_use_uv = fc2_use_uv,qkv_len = qkv_len,proj_len = proj_len, fc1_len = fc1_len, fc2_len = fc2_len)
    teacher_model = VisionTransformer(patch_size=16,  embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True)

    # model = VisionTransformerAda(patch_size=16,  embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True,
    #         qkv_use_uv = qkv_use_uv,proj_use_uv=proj_use_uv,fc1_use_uv = fc1_use_uv,fc2_use_uv = fc2_use_uv,qkv_len = qkv_len,proj_len = proj_len, fc1_len = fc1_len, fc2_len = fc2_len)
    # teacher_model = VisionTransformer(patch_size=16,  embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True)


    # model = VisionTransformerAda(patch_size=16,  embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
    #         qkv_use_uv = qkv_use_uv,fc1_use_uv = fc1_use_uv,fc2_use_uv = fc2_use_uv,qkv_len = qkv_len, fc1_len = fc1_len, fc2_len = fc2_len)
    # teacher_model = VisionTransformer(patch_size=16,  embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True)

    teacher_model_sum = sum(p.numel() for p in teacher_model.state_dict().values())
    model_sum = sum(p.numel() for p in model.state_dict().values())
    if model_sum > teacher_model_sum:
        return

    model.cuda()
    teacher_model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module
    teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    teacher_model_without_ddp = teacher_model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")


    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    # load_path = "deit/ffn_output/all_fc_ada/deit_b/deit_b_fewshot2_"
    # load_path = "deit/ffn_output/all_fc_ada/deit_s/deit_s_fewshot2_"
    load_path = "deit/ffn_output/all_fc_ada/deit_t/deit_t_fewshot2_"
    new_checkpoint = merge_model(checkpoint['model'], args.block_i, args.fc_i, args.dim_i, load_path)


    if 'model' in new_checkpoint.keys():
        msg = model_without_ddp.load_state_dict(new_checkpoint['model'], strict=False)
    else:
        msg = model_without_ddp.load_state_dict(new_checkpoint, strict=False)
    logger.info(msg)

    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    if 'model' in checkpoint.keys():
        msg = teacher_model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    else:
        msg = teacher_model_without_ddp.load_state_dict(checkpoint, strict=False)
    logger.info(msg)


    acc1, acc5, loss = validate(config, data_loader_val, model, teacher_model)
    logger.info(f"Accuracy of the network on the val test images: {acc1:.1f}%")


@torch.no_grad()
def validate(config, data_loader, model, teacher_model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    teacher_model.eval()


    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    kl = AverageMeter()

    end = time.time()


    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)


        output,_ = model(images)
        teacher_output,_ = teacher_model(images)

        logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()
        softmax = torch.nn.Softmax(dim=1).cuda()
        distil_loss = torch.sum(
            torch.sum(softmax(teacher_output) * (logsoftmax(teacher_output)-logsoftmax(output)), dim=1))

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)
        distil_loss = reduce_tensor(distil_loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        kl.update(distil_loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'kl {kl.val:.3f} ({kl.avg:.3f})\t'

                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}  KL {kl.avg:.3f}')

    f_name = "deit/" + str(args.block_i) + "_" + str(args.fc_i) + ".txt"
    with open(f_name, "a") as f:
        f.writelines(str(args.dim_i) + "," + str(kl.avg) + "\n")

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    args = parse_option()
    main(args)
