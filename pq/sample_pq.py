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
import sys
sys.path.append("/mnt/petrelfs/shaojie/code/DiT/")
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from copy import deepcopy
import os

from pqf.compression.model_compression import compress_model
from pqf.utils.config_loader import load_config
from pqf.utils.logging import get_tensorboard_logger, log_compression_ratio, log_config, setup_pretty_logging
from pqf.utils.model_size import compute_model_nbits

_MODEL_OUTPUT_PATH_SUFFIX = "trained_models"


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

    file_path = os.path.dirname(__file__)
    default_config = os.path.join(file_path, "../pqf/config/train_dit_val.yaml")
    config = load_config(file_path, default_config_path=default_config)
    summary_writer = get_tensorboard_logger(config["output_path"])
    log_config(config, summary_writer)
    model_config = config["model"]
    compression_config = model_config["compression_parameters"]

    uncompressed_model_size_bits = compute_model_nbits(model)
    model_pq = compress_model(deepcopy(model), **compression_config).cuda()
    compressed_model_size_bits = compute_model_nbits(model_pq)
    log_compression_ratio(uncompressed_model_size_bits, compressed_model_size_bits, summary_writer)
    
    model_pq.load_state_dict(torch.load("results/002-DiT-XL-2/checkpoints/0005000.pt")['model'])
    model_pq.eval()
    diffusion.forward_val(vae, model.forward_with_cfg, model_pq.forward_with_cfg, cfg=True, name="results/002-DiT-XL-2/sample_cfg")
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
    args = parser.parse_args()
    main(args)

