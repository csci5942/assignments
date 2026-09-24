#!/bin/bash
#SBATCH --job-name=a4-pp-a40
#SBATCH --account=bidb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=60g
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# ---------------------------------------------------------------------
# Part 3's bubble sweep: `large` at pp=4, three micro-batch sizes, both
# schedules.
#
#   sbatch slurm/pp_a40.sbatch --config configs/large.json
#
# **Part 3 uses a global batch of 32, not 128.** The last stage owns the
# output head, so under AFAB it holds `(micro, T, 50304)` logits alive
# for every micro-batch in flight -- and since `m x micro` is the global
# batch, that total is fixed at `global_batch x T x V` however you split
# it. At 128 it is 13.2 GB of logits and about as much again for the
# softmax saved for backward, which does not fit on a 44 GiB A40
# alongside the model. Shrinking the micro-batch does not help: it moves
# memory between `m` and `micro` and leaves the product alone. Shrinking
# the batch is the only lever.
#
# The micro-batch sets `m`, the number of micro-batches per step:
#
#   micro 8 -> m =  4    predicted bubble (p-1)/(m+p-1) = 3/7   = 42.9%
#   micro 4 -> m =  8                                     3/11  = 27.3%
#   micro 2 -> m = 16                                     3/19  = 15.8%
#   micro 1 -> m = 32                                     3/35  =  8.6%
#
# Both schedules run at every `m`, because the comparison that matters is
# not which is faster -- the bubble is identical and they should be
# within noise of each other -- but what each costs in memory. AFAB
# holds all `m` micro-batches' activations alive; 1F1B holds at most `p`.
# Watch the peak memory column as `m` rises.
#
# `--mem=60g` on purpose. 220g would make memory the binding TRES and
# bill this job well above its GPU share.
# ---------------------------------------------------------------------
set -uo pipefail
mkdir -p logs

module load pytorch-conda/2.12
conda activate base

export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
CONFIG=configs/large.json
STEPS=12

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for MICRO in 8 4 2 1; do
  for SCHEDULE in afab 1f1b; do
    echo
    echo "===== micro-batch ${MICRO}, ${SCHEDULE} ====="
    # A configuration that runs out of memory is a *result*, not a
    # failure: AFAB holds every micro-batch's activations alive, so it
    # is expected to die at an `m` where 1F1B is comfortable. Record it
    # and keep going rather than losing the rest of the sweep.
    srun python -m trainer.train \
         --config "${CONFIG}" --steps "${STEPS}" --warmup-steps 3 \
         --pp 4 --dp 1 --ddp none --global-batch 32 \
         --micro-batch "${MICRO}" --schedule "${SCHEDULE}" \
         --tag "pp4-m${MICRO}-${SCHEDULE}" "$@" \
      || echo "  OUT OF MEMORY at micro-batch ${MICRO} under ${SCHEDULE}"
  done
done

echo
echo "===== bubble and memory ====="
python - <<'EOF'
import glob, json
rows = []
for path in sorted(glob.glob("out/large-pp4-*/summary.json")):
    s = json.load(open(path))
    m = s["accum"]
    p = s["pp"]
    rows.append((m, s["schedule"], s["tokens_per_second"],
                 s.get("busy_fraction"), (p - 1) / (m + p - 1),
                 (s.get("memory") or {}).get("measured_peak", 0) / 2**30))
print(f"{'m':>4} {'sched':>6} {'tok/s':>10} {'busy':>8} "
      f"{'predicted bubble':>18} {'measured idle':>15} {'peak GiB':>10}")
for m, sched, tps, busy, pred, mem in sorted(rows):
    idle = (1 - busy) if busy else float("nan")
    print(f"{m:>4} {sched:>6} {tps:>10,.0f} {busy or 0:>8.3f} "
          f"{pred*100:>17.1f}% {idle*100:>14.1f}% {mem:>10.2f}")
EOF
