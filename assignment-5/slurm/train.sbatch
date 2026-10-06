#!/bin/bash
#SBATCH --job-name=a5-train
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
# A4's test.sbatch, pointed at this assignment's train.py.
#   sbatch slurm/train.sbatch --config configs/gpt2-medium-full.json
#   sbatch slurm/train.sbatch --config configs/lora-r4.json --steps 50
# ---------------------------------------------------------------------
set -euo pipefail
mkdir -p logs
module load pytorch-conda/2.12
conda activate base
source .venv/bin/activate          # the venv from Part 0, on top of the module
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun python train.py --config configs/lora-r4.json "$@"
