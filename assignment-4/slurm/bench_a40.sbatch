#!/bin/bash
#SBATCH --job-name=a4-bench-a40
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=60g
# 60g, not a full node's 220g, on purpose. Delta bills the *largest*
# of your CPU, memory and GPU requests: 220g weighs 1760 against a
# single GPU's 500, so a one-GPU job asking for it bills as three and
# a half. 60g weighs 480 and never binds.
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
# Part 0's warm-up: what does it cost these GPUs to talk to each other?
#
#   sbatch slurm/bench_a40.sbatch
#
# Two curves, at two and at four A40s, from 1 MB to 1 GB. About three
# minutes of wall-clock, roughly 0.2 A40 GPU-hours. It is the cheapest
# measurement in the assignment and the one the most later questions
# depend on, so do it before you write any code.
#
# Both runs append to bench/busbw-a40.csv, which is what you plot.
#
# You need one more curve later, and only later: eight GPUs across two
# nodes, in Part 4, where the gap between that curve and this one is
# the number the whole part turns on. It is deferred because it costs
# four times what this does and nothing before Part 4 uses it.
# ---------------------------------------------------------------------
set -euo pipefail
mkdir -p logs bench

module load pytorch-conda/2.12
conda activate base

export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
OUT=bench/busbw-a40.csv
rm -f "${OUT}"

nvidia-smi --query-gpu=name --format=csv,noheader | head -1

for N in 2 4; do
  echo
  echo "===== ${N} A40s ====="
  srun --ntasks="${N}" --gpus="${N}" python -m bench.allreduce \
       --min-mb 1 --max-mb 1024 --csv "${OUT}" "$@"
done

echo
echo "wrote ${OUT}"
echo "Plot busbw_gbps against mb, one line per world_size. Keep the"
echo "plateau value somewhere you can find it -- Parts 1, 2 and 4 all"
echo "ask you to predict a communication time with it."
