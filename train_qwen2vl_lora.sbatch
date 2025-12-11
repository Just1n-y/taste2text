#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --job-name=qwen2vl_lora
#SBATCH --output=logs/qwen2vl_%j.out
#SBATCH --error=logs/qwen2vl_%j.err
#SBATCH --ntasks-per-node=1

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"

# ----------------------------
# ENVIRONMENT STABILITY
# ----------------------------
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=hsn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES=0,1,2,3

# ----------------------------
# MODEL / HF CACHES
# ----------------------------
export HF_HOME=/pscratch/sd/a/abhipa/hf_home
export TRANSFORMERS_CACHE=/pscratch/sd/a/abhipa/hf_home
export HF_DATASETS_CACHE=/pscratch/sd/a/abhipa/hf_home
export TORCH_EXTENSIONS_DIR=/pscratch/sd/a/abhipa/torch_extensions
mkdir -p $HF_HOME logs

# ----------------------------
# ACTIVATE ENV
# ----------------------------
source /global/homes/a/abhipa/miniconda3/etc/profile.d/conda.sh
conda activate /pscratch/sd/a/abhipa/conda-envs/qwen

# ----------------------------
# MASTER ADDRESS SETUP
# ----------------------------
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# ----------------------------
# LAUNCH DISTRIBUTED
# ----------------------------
srun torchrun \
  --nnodes=2 \
  --nproc-per-node=4 \
  --rdzv-backend=c10d \
  --rdzv-id=$SLURM_JOB_ID \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  train_qwen2vl_trl_minimal.py \
    --train-parquet /pscratch/sd/a/abhipa/finalproject/googlelocal_parquet/training_data.parquet \
    --eval-parquet /pscratch/sd/a/abhipa/finalproject/googlelocal_parquet/validation_data.parquet \
    --output-dir /pscratch/sd/a/abhipa/qwen-8gpu-70k \
    --batch-size 1 \
    --grad-accum 8 \
    --epochs 1 