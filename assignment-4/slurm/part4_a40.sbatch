#!/bin/bash
#SBATCH --job-name=a4-part4
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
# `huge` is built on the CPU before it moves to the GPU -- 15 GiB a rank,
# and four ranks to a node, so 60g is not enough and the job dies with a
# Slurm OOM that says nothing about the model. 120g is still free: Delta
# bills the *largest* of your CPU, memory and GPU requests, four GPUs
# weigh 2000, and memory weighs 8 a GB, so anything under 250g a node is
# covered by the GPUs you already asked for.
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
# Part 4, on the two nodes the part actually requires. Everything else
# in slurm/ asks for one node; this is the only script that crosses the
# network, which is the whole subject of Part 4.
#
# It runs ONE configuration, whatever you pass it:
#
#   sbatch slurm/part4_a40.sbatch --dp 8 --tp 1 --pp 1 --ddp overlapped
#   sbatch slurm/part4_a40.sbatch --dp 2 --tp 2 --pp 2 --ddp overlapped \
#          --schedule 1f1b --micro-batch 4 --axis-order tp,dp,pp
#
# Six steps is the measurement the part is scored on; pass
# `--steps 6 --warmup-steps 1`. `huge` takes twelve to forty seconds a
# step depending on how you map it, so a configuration is two to four
# minutes. Submit a separate tagged job for each configuration.
#
# Tag your runs. Each writes out/<config>-<tag>/summary.json, and the
# table of everything you tried is a graded deliverable -- the losing
# configurations are worth as much as the winner, so keep them.
# ---------------------------------------------------------------------
set -euo pipefail
mkdir -p logs

module load pytorch-conda/2.12
conda activate base

export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

echo "nodes: ${SLURM_JOB_NODELIST}"
echo "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-unset}" \
     "FI_PROVIDER=${FI_PROVIDER:-unset}"
srun --ntasks=1 nvidia-smi --query-gpu=name --format=csv,noheader | head -1

srun python -m trainer.train \
     --config configs/huge.json \
     "$@"
