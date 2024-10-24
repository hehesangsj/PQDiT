import os
import argparse
import sys
sys.path.append("/mnt/petrelfs/shaojie/code/DiT/")
import torch
from copy import deepcopy
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from distributed import init_distributed_mode
from torch.nn.parallel import DistributedDataParallel as DDP

from models import DiT_models
from pq.utils_model import init_env, init_model, get_pq_model, cleanup, parse_option
from pq.utils_traineval import sample, train, dit_generator, save_ckpt

def main(args):
    init_distributed_mode(args)
    rank, device, logger, experiment_dir = init_env(args, dir='001-DiT-XL-2')
    model, state_dict, diffusion, vae = init_model(args, device)
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    latent_size = args.image_size // 8

    # pq model
    file_path = os.path.dirname(__file__)
    model_pq = get_pq_model(deepcopy(model), file_path, rank, experiment_dir, logger)
    model_pq = DDP(model_pq.to(device), device_ids=[rank])

    mode = args.pq_mode
    if mode == "sample":
        model_pq.eval()
        model_string_name = args.model.replace("/", "-")
        ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
        folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                    f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
        sample_folder_dir = f"{args.sample_dir}/{folder_name}"
        sample(args, model_pq, vae, diffusion, sample_folder_dir)
    elif mode == "gen":
        model_pq.eval()
        save_ckpt(model_pq, args, checkpoint_dir, logger)
        diffusion_gen = dit_generator('250', latent_size=latent_size, device=device)
        diffusion_gen.forward_val(vae, model.forward, model_pq.forward, cfg=False, name=f"{experiment_dir}/sample_cfg")
    elif mode == "train":
        model_pq.train()
        train(args, logger, model_pq, vae, diffusion, checkpoint_dir)

    logger.info("Done!")
    # cleanup()


if __name__ == "__main__":
    args = parse_option()
    main(args)