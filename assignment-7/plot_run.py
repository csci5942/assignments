"""Reward, length and KL over the steps of one or more runs, and the
before-and-after table from eval.py's results.

    python plot_run.py out/judge-leash                          # writes curves.png
    python plot_run.py out/judge-leash out/judge-noleash out/judge-yours
    python plot_run.py --table results/sft.json results/judge-leash.json

Given.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def curves(run_dirs, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for run in run_dirs:
        with open(os.path.join(run, "log.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        steps = [r["step"] for r in rows]
        name = os.path.basename(os.path.normpath(run))
        axes[0].plot(steps, [r["mean_reward"] for r in rows], label=name)
        axes[1].plot(steps, [r["mean_len"] for r in rows], label=name)
        axes[2].plot(steps, [r["mean_kl"] for r in rows], label=name)
    axes[0].set_ylabel("judge score of the sampled group (1 to 10)")
    axes[1].set_ylabel("completion length (tokens)")
    axes[2].set_ylabel("KL to the start (k3, nats per token)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def table(result_files):
    cols = ("name", "judge_mean", "judge_ci95", "human_mean", "rank_mean", "bleu", "slot_coverage", "mean_output_tokens", "fineweb_val_loss")
    print("| run | judge | +/- | humans | rank of 8 | BLEU | slots | tokens | FineWeb |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for path in result_files:
        with open(path) as f:
            r = json.load(f)
        vals = [r.get(c, "") for c in cols]
        vals[6] = f"{100 * vals[6]:.1f}%" if vals[6] != "" else ""
        print("| " + " | ".join(str(v) for v in vals) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="run directories, or results JSONs with --table")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "curves.png"))
    args = ap.parse_args()
    if args.table:
        table(args.paths)
    else:
        curves(args.paths, args.out)


if __name__ == "__main__":
    main()
