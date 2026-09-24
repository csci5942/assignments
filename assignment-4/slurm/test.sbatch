#!/bin/bash
#SBATCH --job-name=a4-test
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
#   sbatch slurm/test.sbatch
#   sbatch slurm/test.sbatch --config configs/medium.json --steps 50
# ---------------------------------------------------------------------
set -euo pipefail
mkdir -p logs

module load pytorch-conda/2.12
conda activate base

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Defaults first, your arguments last. argparse takes the last occurrence
# of a flag, so anything you pass on the sbatch line overrides what is
# here without you having to edit this file.
srun python -m trainer.train \
     --config configs/small.json \
     --steps 30 --warmup-steps 5 --eval-every 15 \
     "$@"
