#!/bin/bash
#SBATCH --job-name=a4-tp-a40
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
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
# Part 2's race: the same model and the same global batch, split four
# different ways across the same four A40s.
#
#   sbatch slurm/tp_a40.sbatch --config configs/large.json
#
#   1 GPU          the baseline everything is measured against
#   DP=4           four replicas, one all-reduce per step, overlappable
#   TP=4           one model in four pieces, two all-reduces per block,
#                  on the critical path where nothing can hide them
#   DP=2 x TP=2    both, which is the point of having a mesh
#
# All four run in one job so the numbers you compare came from the same
# four GPUs on the same node. Each writes its own out/<name>-<tag>/, so
# the summaries survive for the table.
#
# The one-GPU run holds the whole four-GPU allocation while it plods
# through the same tokens, which is wasteful -- but splitting it into a
# separate job means a separate queue wait and a different set of GPUs,
# and for a comparison this tight that is the worse trade.
# ---------------------------------------------------------------------
set -euo pipefail
mkdir -p logs

module load pytorch-conda/2.12
conda activate base

export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
CONFIG=configs/large.json
STEPS=25
WARMUP=5

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "===== 1 GPU ====="
srun --ntasks=1 --gpus=1 python -m trainer.train \
     --config "${CONFIG}" --steps "${STEPS}" --warmup-steps "${WARMUP}" \
     --ddp none --tag tp-x1 "$@"

echo
echo "===== DP=4 ====="
srun python -m trainer.train \
     --config "${CONFIG}" --steps "${STEPS}" --warmup-steps "${WARMUP}" \
     --dp 4 --tp 1 --ddp overlapped --tag tp-dp4 "$@"

echo
echo "===== TP=4 ====="
srun python -m trainer.train \
     --config "${CONFIG}" --steps "${STEPS}" --warmup-steps "${WARMUP}" \
     --dp 1 --tp 4 --ddp none --tag tp-tp4 "$@"

echo
echo "===== DP=2 x TP=2 ====="
srun python -m trainer.train \
     --config "${CONFIG}" --steps "${STEPS}" --warmup-steps "${WARMUP}" \
     --dp 2 --tp 2 --ddp overlapped --tag tp-dp2tp2 "$@"

echo
echo "===== summaries ====="
for d in out/*tp-*/summary.json; do
  python - "$d" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
mem = s.get("memory") or {}
print(f"  {s['name']:<22} dp={s['dp']} tp={s['tp']}  "
      f"{s['tokens_per_second']:>9,.0f} tok/s  "
      f"{(s.get('mfu') or 0)*100:5.1f}% MFU  "
      f"{mem.get('measured_peak', 0)/2**30:6.2f} GiB peak")
EOF
done
