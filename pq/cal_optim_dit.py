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

import sys
sys.path.append("/mnt/petrelfs/shaojie/code/DiT/")
from models import DiT_models
from pq.main_dit_calkl_s2_1 import reset_param, parse_option, merge_model
from download import find_model
from pq.low_rank_models import DiT_uv_models
from diffusers.models import AutoencoderKL
from pq.sample_pq import dit_generator
from distributed import init_distributed_mode
from pq.main_dit_calkl_s2 import create_logger, center_crop_arr, cleanup


def dp(value, speed):
    # [idx_dim, idx_block]: loss
    # speed: #param
    count = 0
    results = {}
    for bi in range(value.shape[1]):  # idx_block
        if results == {}:  # init
            for bj in range(value.shape[0]):  # idx_dim
                results[speed[bj]] = [bj, value[bj, bi]]  # results: #param, [idx_dim, loss]
        else:
            single = {}
            for k, v in results.items():  # param, [idx_dim, loss]
                for bj in range(value.shape[0]):  # idx_dim
                    sum_val =  v[-1] + value[bj, bi]  # former loss + loss
                    count = float(k) + speed[bj]  # former param + param
                    if count not in single.keys():
                        single[count] = v[:-1] + [bj, sum_val]  # #param, [idx_dim1, idx_dim2, loss]
                    elif sum_val < single[count][-1]:
                        single[count] = v[:-1] + [bj, sum_val]  # update the optimized path
            results = single  # results: # param, [idx_dim1, idx_dim2, ..., loss]
    return results

def remove_unreason_item(results):
    f = {}
    keys = list(results.keys())  # #param
    keys.sort(reverse=False)  # from small to large
    b = {key: results[key] for key in keys}
    for k, v in b.items():  # #param, [path, loss]
        if list(f.keys()) == []:
            f[k] = v 
        else:
            flag = True
            for fk, fv in f.items():
                if fv[-1] < v[-1]:
                    flag = False
            if flag:
                f[k] = v  # remove abnormal value: less compressed weight, but larger value
    return f

def get_compress_results(params, dim, value, f):
    for b in range(28):
        ind_value = b
        file_name = "results/low_rank/008-DiT-XL-2/dim_loss/"+ str(b) + "_" +str(f) + ".txt"
        files = np.loadtxt(file_name,delimiter=',')
        param_kl_dict = dict(zip(files[:,0].astype(np.int32), files[:,1]))  # dim, loss
        for i in range(len(dim)):
            if dim[i] in param_kl_dict.keys():
                value[i, ind_value] = param_kl_dict[dim[i]]  # [idx_dim, idx_block]: loss
        value[0] = 0  # not pq
    results = dp(value, params)
    results = remove_unreason_item(results)
    return results

def get_ind(percent, results):
    params = [key for key in results]
    total_param = params[-1]  # Total parameters from the last key
    target_param_count = total_param * percent
    closest_key = min(params, key=lambda x: abs(x - target_param_count))
    return results[closest_key]
    
def get_dims(ind, blocks, dim):
    uv_dim = []
    use_uv = [True] * blocks
    for i in range(blocks):
        uv_dim.append(dim[ind[i]])
        if ind[i] == 0:
            use_uv[i] = False
    return uv_dim, use_uv

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

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/009-DiT-XL-2"  # Create an experiment folder
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
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    load_path = "results/low_rank/002-DiT-XL-2/dit_t_in_"

    # DiT-XL/2-256
    fc1_params = []
    fc1_dim = [921.6] + np.arange(896, 32, -32).tolist()

    bias = to_2tuple(True)
    in_features = 1152
    hidden_features = 4608
    for dim_idx in range(len(fc1_dim)):
        if dim_idx != 0:
            fc1_u = nn.Linear(in_features, fc1_dim[dim_idx], bias=bias[0])
            n_parameters_u = sum(p.numel() for p in fc1_u.parameters())
            fc1_v = nn.Linear(fc1_dim[dim_idx], hidden_features, bias=bias[0])
            n_parameters_v = sum(p.numel() for p in fc1_v.parameters())
            n_parameters = n_parameters_u + n_parameters_v
            del fc1_u, fc1_v
        else:
            fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
            n_parameters = sum(p.numel() for p in fc1.parameters())
            del fc1
        fc1_params.append(n_parameters)

    fc1_value = np.zeros([len(fc1_dim), 28])
    results_fc1 = get_compress_results(fc1_params, fc1_dim, fc1_value, f=1)
    ind = get_ind(percent=0.5, results=results_fc1)
    fc1_len, fc2_len, qkv_len, proj_len, adaln_len, fc1_use_uv, fc2_use_uv, qkv_use_uv, proj_use_uv, adaln_use_uv = reset_param(28)
    fc1_len, fc1_use_uv = get_dims(ind, 28, fc1_dim)

    model_uv = DiT_uv_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        qkv_use_uv=qkv_use_uv, proj_use_uv=proj_use_uv, fc1_use_uv=fc1_use_uv, fc2_use_uv=fc2_use_uv, adaln_use_uv=adaln_use_uv,
        qkv_len=qkv_len, proj_len=proj_len, fc1_len=fc1_len, fc2_len=fc2_len, adaln_len=adaln_len
    ).to(device)
    state_dict_merge = deepcopy(state_dict)
    for fc_i in range(1, 2):
        for block_i in range(28):
            if fc1_use_uv[block_i] == True:
                state_dict_merge = merge_model(state_dict_merge, block_i, fc_i, fc1_len[block_i], load_path, logger)
    msg = model_uv.load_state_dict(state_dict_merge)
    logger.info(msg)
    model_uv.eval()
    diffusion.forward_val(vae, model.forward, model_uv.forward, cfg=False, name=experiment_dir+"/sample_fc1_0_66_cfg")

if __name__ == "__main__":
    args = parse_option()
    main(args)