#!/bin/bash
set -x

CONFIG=${1}

GPUS=${2:-8}
GPUS_PER_NODE=${3:-8}
# GPUS=${2:-1}
# GPUS_PER_NODE=${3:-1}
PARTITION=${4:-"INTERN3"}
QUOTA_TYPE=${5:-"reserved"}
JOB_NAME=${6:-"vl_sj"}

CPUS_PER_TASK=${CPUS_PER_TASK:-10}

if [ $GPUS -lt 8 ]; then
    NODES=1
else
    NODES=$((GPUS / GPUS_PER_NODE))
fi

SRUN_ARGS=${SRUN_ARGS:-" --jobid=3709386"} # 3768157 3768158 3789766 -w HOST-10-140-66-41  3636795

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=32426    
# export NCCL_DEBUG=INFO
# export TF_CPP_MIN_LOG_LEVEL=3
# unset CUDA_LAUNCH_BLOCKING
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_LOGS="+dynamo" 
# export TORCHDYNAMO_VERBOSE=1
# export TORCH_DISTRIBUTED_DEBUG=INFO

# enable for sampling only
# export LD_LIBRARY_PATH="/mnt/petrelfs/share_data/tianchangyao.p/cuda/cuda-11.7/lib64":$LD_LIBRARY_PATH

srun -p ${PARTITION} \
  --job-name=${JOB_NAME} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python pq/pq_getmodel.py --image-size 256 --ckpt pretrained_models/DiT-XL-2-256x256.pt --pq-mode train --results-dir results/train_pq --global-batch-size 256 --epochs 1 --ckpt-every 5000 --data-path /mnt/petrelfs/share/images/train
#   python pq/pq_getmodel.py --image-size 256 --ckpt pretrained_models/DiT-XL-2-256x256.pt --pq-mode gen --results-dir results/train_pq
#   python pq/cal_optim_dit.py --num-fid-samples 50000 --ckpt pretrained_models/DiT-XL-2-256x256.pt --results-dir samples
#   python evaluator.py samples/VIRTUAL_imagenet256_labeled.npz samples/009-DiT-XL-2/sample_allfc12345_0_8.npz

