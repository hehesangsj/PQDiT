import numpy as np
from glob import glob
from copy import deepcopy
import torch
from timm.layers.helpers import to_2tuple
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import sys
sys.path.append("/mnt/petrelfs/shaojie/code/DiT/")
from models import DiT_models
from distributed import init_distributed_mode
from pq.low_rank_models import DiT_uv_models
from pq.utils_model import parse_option, init_env, init_model, get_pq_model, log_params, log_compare_weights, vis_weights
from pq.utils_traineval import sample, dit_generator, train, save_ckpt
from pq.utils_smoothquant import smooth_dit
from pq.low_rank_compress import get_blocks, merge_model


def main(args):
    init_distributed_mode(args)
    rank, device, logger, experiment_dir = init_env(args)
    # rank, device, logger, experiment_dir = init_env(args, dir='011-DiT-XL-2')
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints

    model, state_dict, diffusion, vae = init_model(args, device)
    # vis_weights(model, logger, f"{experiment_dir}/image_weights", mode='original')
    latent_size = args.image_size // 8
    load_path = "results/low_rank/002-DiT-XL-2/dit_t_in_"
    percent = args.percent
    fc_space = range(1, 6)
    fc_len, fc_use_uv = {}, {}
    block_str = ''.join(map(str, fc_space))

    for fc_idx in range(1, 6):
        if fc_idx in fc_space:
            fc_len[fc_idx], fc_use_uv[fc_idx] = get_blocks(percent, fc_idx, compress_all=True)
        else:
            fc_len[fc_idx] = [128] * 28
            fc_use_uv[fc_idx] = [False] * 28
        # fc_len[fc_idx] = [128] * 28
        # fc_use_uv[fc_idx] = [False] * 28

    model_uv = DiT_uv_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        qkv_use_uv=fc_use_uv[3], proj_use_uv=fc_use_uv[4], 
        fc1_use_uv=fc_use_uv[1], fc2_use_uv=fc_use_uv[2], adaln_use_uv=fc_use_uv[5],
        qkv_len=fc_len[3], proj_len=fc_len[4], 
        fc1_len=fc_len[1], fc2_len=fc_len[2], adaln_len=fc_len[5]
    ).to(device)

    if args.low_rank_ckpt == None:
        state_dict_merge = deepcopy(state_dict)
        # Iterate through blocks and merge model
        for fc_i in fc_space:
            for block_i in range(28):
                if fc_use_uv[fc_i][block_i]:
                    state_dict_merge = merge_model(state_dict_merge, block_i, fc_i, fc_len[fc_i][block_i], load_path, logger)                
        msg = model_uv.load_state_dict(state_dict_merge)
        checkpoint_dir_lowrank = f"{experiment_dir}/checkpoints-low-rank"  # Stores saved model checkpoints
        save_ckpt(model_uv, args, checkpoint_dir_lowrank, logger)
        logger.info(msg)
    else:
        model_uv.load_state_dict(torch.load(args.low_rank_ckpt)['model'])
    log_params(model, model_uv, logger)
    logger.info("Low rank compression done!")
    log_compare_weights(model_comp=model_uv, model_orig=model, compress_mode='uv', logger=logger)
    # image_weight_dir = f"{experiment_dir}/image_weights_uv"
    # vis_weights(model_uv, logger, image_weight_dir)

    if args.smooth:
        smooth_dit(model_uv)
        logger.info("Smooth quant done!")
        log_compare_weights(model_comp=model_uv, model_orig=model, compress_mode='uv', logger=logger)

    # image_weight_dir = f"{experiment_dir}/new_images/image_weights_uv_smooth"
    # vis_weights(model_uv, logger, image_weight_dir)

    # image_name = f"sample_allfc{block_str}_{percent:.1f}".replace('.', '_')
    # diffusion_gen = dit_generator('250', latent_size=latent_size, device=device)
    # diffusion_gen.forward_val(vae, model.forward, model_uv.forward, cfg=False, name=f"{experiment_dir}/{image_name}", logger=logger)
    # return
    
    if args.pq_after_low_rank:
        if args.pq_ckpt == None:
            file_path = os.path.dirname(__file__)
            model_uv = get_pq_model(model_uv, file_path, rank, experiment_dir, logger, mode='train')
            logger.info("PQ model done!")
            log_compare_weights(model_comp=model_uv, model_orig=model, compress_mode='pq', logger=logger)
            # checkpoint_dir_pq = f"{experiment_dir}/checkpoints-pq-smooth"  # Stores saved model checkpoints
            # save_ckpt(model_uv, args, checkpoint_dir_pq, logger)
        else:
            file_path = os.path.dirname(__file__)
            model_uv = get_pq_model(model_uv, file_path, rank, experiment_dir, logger, mode='val')
            model_uv.load_state_dict(torch.load(args.pq_ckpt)['model'])
    log_params(model, model_uv, logger)

    model_uv = DDP(model_uv.to(device), device_ids=[rank])
    mode = args.low_rank_mode
    if mode == "sample":
        model_uv.eval()
        image_name = f"sample_allfc{block_str}_{percent:.1f}".replace('.', '_')
        image_dir = f"{experiment_dir}/{image_name}"
        sample(args, model_uv, vae, diffusion, image_dir)
    elif mode == "gen":
        model_uv.eval()
        diffusion_gen = dit_generator('250', latent_size=latent_size, device=device)
        image_name = f"images_new/sample_allfc{block_str}_{percent:.1f}_pq".replace('.', '_')
        diffusion_gen.forward_val(vae, model.forward, model_uv.forward, cfg=False, name=f"{experiment_dir}/{image_name}", logger=logger)
    elif mode == "train":
        model_uv.train()
        train(args, logger, model_uv, vae, diffusion, checkpoint_dir)

if __name__ == "__main__":
    args = parse_option()
    main(args)
    