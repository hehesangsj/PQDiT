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
    """
    Dynamic programming method to compute an optimized path through the dimensions and blocks, 
    minimizing the loss while taking into account the parameter cost.

    Args:
        value (np.ndarray): A 2D array of shape [num_dims, num_blocks], where value[i, j] represents 
                            the loss for the i-th dimension in the j-th block.
        speed (list or np.ndarray): A 1D array where each element represents the parameter cost for 
                                    each dimension.

    Returns:
        dict: A dictionary where the keys are the total number of parameters (sum of selected dimensions' 
              parameter costs), and the values are lists of the selected dimensions for each block and 
              the total accumulated loss.
              Format: {total_param_count: [dim1_idx, dim2_idx, ..., total_loss]}

    Logic:
        - The function goes through all blocks, and for each block, it tries to find the optimal combination
          of dimensions that minimizes the loss while summing up the parameter cost.
        - It starts by initializing with the first block and progressively updates the results by checking
          the sum of losses and parameter costs for subsequent blocks.
    """

    count = 0 
    results = {}

    for bi in range(value.shape[1]):
        if results == {}:
            for bj in range(value.shape[0]):
                results[speed[bj]] = [bj, value[bj, bi]]  # results: {param_cost: [dim_idx, loss]}
        else:
            single = {}
            for k, v in results.items():  # k: param_cost, v: [dim_idx_list, loss]
                # Iterate over the dimensions in the current block
                for bj in range(value.shape[0]):
                    sum_val = v[-1] + value[bj, bi]  # Calculate the new loss (previous loss + current loss)
                    count = float(k) + speed[bj]  # Calculate the new parameter cost (previous + current)
                    # Update the results only if this path leads to a smaller loss for the same param count
                    if count not in single.keys():
                        single[count] = v[:-1] + [bj, sum_val]  # New entry with the new dimension and loss
                    elif sum_val < single[count][-1]:
                        single[count] = v[:-1] + [bj, sum_val]  # Update with the better path
            results = single

    return results


def remove_unreason_item(results):
    """
    Filters out suboptimal paths from the results, removing cases where higher parameter cost 
    results in a higher loss. Keeps only the paths where the loss decreases as the parameter cost increases.

    Args:
        results (dict): A dictionary where keys are parameter counts and values are lists 
                        containing the path of selected dimensions and the associated loss.
                        Format: {param_count: [dim1_idx, dim2_idx, ..., total_loss]}

    Returns:
        dict: A filtered dictionary where each key is a parameter count, and the value is the 
              path and total loss for that param count. Only paths with a decreasing loss for increasing 
              parameter costs are retained.
    """

    f = {}
    keys = list(results.keys())
    keys.sort(reverse=False)  # Sort parameter counts in ascending order
    b = {key: results[key] for key in keys}

    # Iterate through the sorted results to filter out suboptimal ones
    for k, v in b.items():  # k: param_count, v: [path, total_loss]
        if not f:
            f[k] = v
        else:
            flag = True
            # Compare the current loss with the losses of already filtered items
            for fk, fv in f.items():
                if fv[-1] < v[-1]:  # If any existing path has a lower loss, don't add this one
                    flag = False
            if flag:
                f[k] = v  # Keep only paths with better (lower) loss

    return f


def get_compress_results(params, dim, value, f):
    """
    Processes compression results by reading loss values from files, populating the 'value' array 
    with the corresponding losses, and then computing and filtering the results.

    Args:
        params (list or np.ndarray): A 1D array of parameter costs for each dimension.
        dim (list or np.ndarray): A list of dimension indices to be checked against the loaded file data.
        value (np.ndarray): A 2D array to store the losses. Shape: [num_dims, num_blocks]. 
                            This will be populated with loss values from the files.
        f (int): A specific file identifier used in constructing file paths for loading the loss data.

    Returns:
        dict: A dictionary containing the optimal paths after dynamic programming and filtering.
              Format: {param_count: [dim1_idx, dim2_idx, ..., total_loss]}
    """

    for b in range(28):
        ind_value = b
        file_name = "results/low_rank/008-DiT-XL-2/dim_loss/" + str(b) + "_" + str(f) + ".txt"
        # Load the loss data from file (expected format: two columns, dimension and loss)
        files = np.loadtxt(file_name, delimiter=',')
        param_kl_dict = dict(zip(files[:, 0].astype(np.int32), files[:, 1]))

        # Iterate through the dimensions and populate the loss values in the 'value' array
        for i in range(len(dim)):
            if dim[i] in param_kl_dict.keys():
                value[i, ind_value] = param_kl_dict[dim[i]]  # Store loss for [dimension, block]

    # Perform dynamic programming to get the optimal paths
    results = dp(value, params)
    # Filter out suboptimal paths
    results = remove_unreason_item(results)

    return results

def get_ind(percent, results, total_param):
    """
    Finds the closest path in the results based on a target percentage of total parameters.

    Args:
        percent (float): The percentage (0 to 1) of the total parameters to be used.
        results (dict): A dictionary where the keys are parameter counts and the values are lists 
                        of selected dimensions and their associated loss.
                        Format: {param_count: [dim1_idx, dim2_idx, ..., total_loss]}
        total_param (float): The total available parameter count.

    Returns:
        list: The optimal path (list of dimension indices) corresponding to the closest 
              parameter count to the target (total_param * percent).
    """

    params = [key for key in results]
    target_param_count = total_param * percent
    # Find the key (parameter count) in 'results' that is closest to the target parameter count
    closest_key = min(params, key=lambda x: abs(x - target_param_count))

    return results[closest_key]  # Return the optimal path corresponding to the closest parameter count

def get_dims(ind, blocks, dim, compress_all=False):
    """
    Extracts the dimensions and usage flags based on the selected indices.

    Args:
        ind (list): A list of indices representing the selected dimensions for each block.
        blocks (int): The number of blocks in the model or structure.
        dim (list or np.ndarray): A list or array containing all available dimension indices.

    Returns:
        tuple:
            uv_dim (list): A list of selected dimensions for each block.
            use_uv (list): A list of boolean flags indicating whether a UV operation is used for each block.
                           If a dimension index is 0, the corresponding flag is set to False.
    """

    uv_dim = []
    use_uv = [True] * blocks

    for i in range(blocks):
        uv_dim.append(dim[ind[i]])  # Append the dimension corresponding to the current index
        if ind[i] == 0 and (not compress_all) :  # If the selected index is 0, set the flag to False for that block
            use_uv[i] = False

    return uv_dim, use_uv


def get_blocks(percent=0.9, f=1, compress_all=False):
    bias = to_2tuple(True)
    dim_ranges = {
        1: ([921.6] + np.arange(896, 32, -32).tolist(), 1152, 4608),
        2: ([921.6] + np.arange(896, 32, -32).tolist(), 4608, 1152),
        3: ([864] + np.arange(832, 32, -32).tolist(), 1152, 3456),
        4: ([576] + np.arange(544, 32, -32).tolist(), 1152, 1152),
        5: ([987.4] + np.arange(960, 32, -32).tolist(), 1152, 6912),
    }

    dim, in_features, hidden_features = dim_ranges[f]
    if compress_all:
        dim = dim[1:]  # Remove the first element if compressing all
    
    params = []
    if compress_all:
        for dim_size in dim:
            param_count_u = in_features * dim_size + (dim_size if bias[0] else 0)
            param_count_v = dim_size * hidden_features + (hidden_features if bias[0] else 0)
            params.append(param_count_u + param_count_v)
    else:
        param_count_fc = in_features * hidden_features + (hidden_features if bias[0] else 0)
        params.append(param_count_fc)
        for dim_size in dim[1:]:
            param_count_u = in_features * dim_size + (dim_size if bias[0] else 0)
            param_count_v = dim_size * hidden_features + (hidden_features if bias[0] else 0)
            params.append(param_count_u + param_count_v)

    total_param = (in_features * hidden_features + (hidden_features if bias[0] else 0)) * 28
    value = np.zeros([len(dim), 28])
    results = get_compress_results(params, dim, value, f=f)
    ind = get_ind(percent=percent, results=results, total_param=total_param)
    
    return get_dims(ind, 28, dim, compress_all)


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
    percent = 0.5
    fc_space = range(1, 6)
    fc_len, fc_use_uv = {}, {}

    # Populate lengths and use_uv flags for each block
    for fc_idx in fc_space:
        fc_len[fc_idx], fc_use_uv[fc_idx] = get_blocks(percent, fc_idx, compress_all=True)

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

                
    msg = model_uv.load_state_dict(state_dict_merge)
    logger.info(msg)
    model_uv.eval()

    block_str = ''.join(map(str, fc_space))
    image_name = f"sample_allfc{block_str}_{percent:.1f}".replace('.', '_')
    diffusion.forward_val(vae, model.forward, model_uv.forward, cfg=False, name=f"{experiment_dir}/{image_name}")

if __name__ == "__main__":
    args = parse_option()
    main(args)