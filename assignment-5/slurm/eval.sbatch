#!/bin/bash
#SBATCH --job-name=a5-eval
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
#   sbatch slurm/eval.sbatch --run out/gpt2-medium-full
#   sbatch slurm/eval.sbatch --base gpt2-medium
# ---------------------------------------------------------------------
set -euo pipefail
mkdir -p logs
module load pytorch-conda/2.12
conda activate base
source .venv/bin/activate
srun python eval.py "$@"
