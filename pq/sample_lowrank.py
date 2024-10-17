# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
import sys
sys.path.append("/mnt/petrelfs/shaojie/code/DiT/")
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from os.path import dirname

from diffusion.gaussian_diffusion import _extract_into_tensor
import diffusion.gaussian_diffusion as gd
from diffusion.respace import space_timesteps

from tqdm.auto import tqdm
from pq.low_rank_models import DiT_uv_models
from pq.sample_pq import dit_generator

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    diffusion = dit_generator(str(args.num_sampling_steps), latent_size=latent_size, device=device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    # with open("model_structure.txt", "w") as f:
    #     print(model, file=f)
    # with open("vae_structure.txt", "w") as f:
    #     print(vae, file=f)

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
    
    ckpt_uv_path = args.ckpt_lr
    ckpt_uv_folder = dirname(dirname(ckpt_uv_path))
    model_uv.load_state_dict(torch.load(ckpt_uv_path)['model'])
    model_uv.eval()
    diffusion.forward_val(vae, model.forward_with_cfg, model_uv.forward_with_cfg, cfg=True, name=ckpt_uv_folder+"/sample_cfg")
    # diffusion.forward_val(vae, model.forward, model_pq.forward, cfg=False, name="sample")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--ckpt-lr", type=str, default=None)
    args = parser.parse_args()
    main(args)


