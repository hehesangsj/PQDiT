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
from copy import deepcopy
import matplotlib.pyplot as plt

from diffusion.gaussian_diffusion import _extract_into_tensor
import diffusion.gaussian_diffusion as gd
from diffusion.respace import space_timesteps

import os
from tqdm.auto import tqdm
import random

from pqf.compression.model_compression import compress_model
from pqf.training.dit_utils import DiTTrainer
from pqf.training.lr_scheduler import get_learning_rate_scheduler
from pqf.training.optimizer import get_optimizer
from pqf.training.training import TrainingLogger, train_one_epoch, train_one_epoch_dit
from pqf.utils.config_loader import load_config
from pqf.utils.logging import get_tensorboard_logger, log_compression_ratio, log_config, setup_pretty_logging
from pqf.utils.model_size import compute_model_nbits
from pqf.utils.state_dict_utils import save_state_dict_compressed

_MODEL_OUTPUT_PATH_SUFFIX = "trained_models"

class dit_generator:
    def __init__(self, timestep_respacing, latent_size, device):
        # create_diffusion
        betas = gd.get_named_beta_schedule('linear', 1000)
        use_timesteps=space_timesteps(1000, timestep_respacing)

        # SpacedDiffusion
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(betas)
        last_alpha_cumprod = 1.0
        new_betas = []
        self.set_alpha(betas)
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        self.set_alpha(new_betas)
        betas = np.array(new_betas, dtype=np.float64)

        # GaussianDiffusion
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        self.num_timesteps = int(betas.shape[0])
        self.latent_size = latent_size
        self.device = device

    def set_alpha(self, betas):
        # GaussianDiffusion
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

    def forward_val(self, vae, model, model_pq, cfg=False, name="sample_pq", save=False):
        # sample
        class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
        z, model_kwargs = self.pre_process(class_labels, cfg=cfg)

        # p_sample_loop_progressive
        img = z
        img_pq = z
        indices = list(range(self.num_timesteps))[::-1]
        indices_tqdm = tqdm(indices)
        mse_loss = []
        for i in indices_tqdm:
            t = torch.tensor([i] * z.shape[0], device=self.device)
            with torch.no_grad():
                # SpacedDiffusion
                if i == 1:
                    print("0")
                map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
                new_ts = map_tensor[t]
                # p_mean_variance
                # model_output = model(img, new_ts, **model_kwargs)
                # model_output_pq = model_pq(img_pq, new_ts, **model_kwargs)

                model_output, feat = model(img, new_ts, **model_kwargs, distill=True)
                model_output_pq, feat_pq = model_pq(img_pq, new_ts, **model_kwargs, distill=True)

                img, img_pq = self.post_process(t, img, img_pq, model_output, model_output_pq)

                mse_loss.append(torch.mean((model_output - model_output_pq) ** 2).cpu().numpy())

                if save:
                    with open('mse_and_mav.csv', 'a') as file:
                        file.write(f"{i},{torch.mean((model_output - model_output_pq) ** 2)},{model_output.abs().mean()}\n")                    

        samples = img    
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        samples_pq = img_pq
        samples_pq, _ = samples_pq.chunk(2, dim=0)  # Remove null class samples
        samples_pq = vae.decode(samples_pq / 0.18215).sample

        if save:
            plt.figure(figsize=(10, 5))
            plt.plot(indices[::-1], mse_loss, label='MSE Loss')
            plt.xlabel('Time Step')
            plt.ylabel('MSE Loss')
            plt.title('MSE Loss Over Time Steps')
            plt.legend()
            plt.grid(True)
            plt.savefig('mse_loss_over_time_steps.png')
                    
            save_image(samples, name+'.png', nrow=4, normalize=True, value_range=(-1, 1))
            save_image(samples_pq, name+'_pq.png', nrow=4, normalize=True, value_range=(-1, 1))

    def pre_process(self, class_labels, cfg=False):
        n = len(class_labels)
        z = torch.randn(n, 4, self.latent_size, self.latent_size, device=self.device)
        y = torch.tensor(class_labels, device=self.device)
        if cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=self.device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            model_kwargs = dict(y=y)
        return z, model_kwargs


    def post_process(self, t, img, img_pq, model_output, model_output_pq):
        # p_mean_variance
        B, C = img.shape[:2]
        model_output, model_var_values = torch.split(model_output, C, dim=1)
        min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, img.shape)
        max_log = _extract_into_tensor(np.log(self.betas), t, img.shape)
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        pred_xstart = self._predict_xstart_from_eps(x_t=img, t=t, eps=model_output)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=img, t=t)

        model_output_pq, model_var_values_pq = torch.split(model_output_pq, C, dim=1)
        frac_pq = (model_var_values_pq + 1) / 2
        model_log_variance_pq = frac_pq * max_log + (1 - frac_pq) * min_log
        pred_xstart_pq = self._predict_xstart_from_eps(x_t=img_pq, t=t, eps=model_output_pq)
        model_mean_pq, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart_pq, x_t=img_pq, t=t)

        # p_sample
        noise = torch.randn_like(img)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
        )  # no noise when t == 0
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

        sample_pq = model_mean_pq + nonzero_mask * torch.exp(0.5 * model_log_variance_pq) * noise

        return sample, sample_pq
    

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


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
    default_config = os.path.join(file_path, "../pqf/config/train_dit.yaml")
    config = load_config(file_path, default_config_path=default_config)
    summary_writer = get_tensorboard_logger(config["output_path"])
    log_config(config, summary_writer)
    model_config = config["model"]
    compression_config = model_config["compression_parameters"]

    uncompressed_model_size_bits = compute_model_nbits(model)
    model_pq = compress_model(deepcopy(model), **compression_config).cuda()
    compressed_model_size_bits = compute_model_nbits(model_pq)
    log_compression_ratio(uncompressed_model_size_bits, compressed_model_size_bits, summary_writer)
    
    model_pq.load_state_dict(torch.load("results/001-DiT-XL-2/checkpoints/0005000.pt")['model'])
    model_pq.eval()
    diffusion.forward_val(vae, model.forward_with_cfg, model_pq.forward_with_cfg, cfg=True, name="sample_cfg")
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


