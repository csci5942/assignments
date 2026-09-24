#!/bin/bash
#SBATCH --job-name=a4-scale-a40
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=60g
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
# Part 1's scaling table, on A40. One GPU count per job; both the naive
# and the overlapped reduction inside it, so the two numbers you compare
# come from the same allocation and the same GPUs.
#
#   sbatch slurm/scale_a40.sbatch --config configs/medium.json
#
# That runs at 4 GPUs, the default above. For the other two points,
# override the allocation on the sbatch line -- Slurm directives in this
# file are defaults.
#
#   sbatch --gpus-per-node=1 --ntasks-per-node=1 slurm/scale_a40.sbatch \
#          --config configs/medium.json
#   sbatch --gpus-per-node=2 --ntasks-per-node=2 slurm/scale_a40.sbatch \
#          --config configs/medium.json
#   ...
#
# Ask for one job per GPU count rather than holding four GPUs while a
# one-GPU run plods through the same tokens. 
#
# ---------------------------------------------------------------------
set -euo pipefail
mkdir -p logs

module load pytorch-conda/2.12
conda activate base

GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_NTASKS:-1}}"
# Two jobs landing on one node must not pick the same rendezvous port.
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "== scaling at ${GPUS} GPU(s), A40 =="

for STRATEGY in naive overlapped; do
  echo
  echo "-- ${STRATEGY}, ${GPUS} GPU(s) --"
  srun python -m trainer.train \
       --config configs/medium.json \
       --steps 25 --warmup-steps 5 \
       --dp "${GPUS}" --ddp "${STRATEGY}" \
       --tag "a40-${STRATEGY}-x${GPUS}" \
       "$@"
done

echo
echo "summaries:"
ls -1 out/*a40*/summary.json 2>/dev/null || true
