#!/bin/bash
set -x

CONFIG=${1}

# GPUS=${2:-8}
# GPUS_PER_NODE=${3:-8}
GPUS=${2:-1}
GPUS_PER_NODE=${3:-1}
PARTITION=${4:-"INTERN3"}
QUOTA_TYPE=${5:-"reserved"}
JOB_NAME=${6:-"vl_sj"}

CPUS_PER_TASK=${CPUS_PER_TASK:-10}

if [ $GPUS -lt 8 ]; then
    NODES=1
else
    NODES=$((GPUS / GPUS_PER_NODE))
fi

# SRUN_ARGS=${SRUN_ARGS:-" --jobid=3722476"} # 3768157 3768158 3789766 -w HOST-10-140-66-41  3636795

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=32424    

QUANT_FLAGS="--image-size 256 --ckpt pretrained_models/DiT-XL-2-256x256.pt \
             --pq \
             --s3-mode gen --global-batch-size 32 \
             --low-rank \
             --low-rank-ckpt results/low_rank/009-DiT-XL-2/checkpoints-low-rank/ckpt.pt \
             --results-dir results/low_rank"
            #  --smooth \
            #  --pq-ckpt results/low_rank/011-DiT-XL-2/checkpoints-pq/ckpt.pt \

SAMPLE_FLAGS="--epochs 100 --ckpt-every 5000 --data-path /mnt/petrelfs/share/images/train"

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
  python -u pq/low_rank_s3_getmodel.py $QUANT_FLAGS $SAMPLE_FLAGS
