"""The rank knee: dev loss and BLEU against trainable parameters, with full
fine-tuning as the reference line. Reads out/*/summary.json and
results/*.json; writes knee.png. Given.

    python plot_knee.py
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(pattern):
    rows = {}
    for path in glob.glob(os.path.join(HERE, pattern)):
        with open(path) as f:
            d = json.load(f)
        rows[d["name"]] = d
    return rows


summaries = load("out/*/summary.json")
results = load("results/*.json")
lora_runs = sorted((s for s in summaries.values()
                    if s["config"]["mode"] == "lora" and s["name"].startswith("lora-r")),
                   key=lambda s: s["trainable"])
full = summaries.get("gpt2-medium-full")
if not lora_runs or full is None:
    raise SystemExit("need out/lora-r*/summary.json and out/gpt2-medium-full/summary.json")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
xs = [s["trainable"] for s in lora_runs]
axes[0].plot(xs, [s["dev_loss"] for s in lora_runs], marker="o", label="LoRA")
axes[0].axhline(full["dev_loss"], color="C3", ls="--", label=f"full fine-tuning, {full['trainable'] / 1e6:.0f}M")
axes[0].set_ylabel("dev loss (nats per scored token)")
for s in lora_runs:
    axes[0].annotate(f"r={s['config']['rank']}", (s["trainable"], s["dev_loss"]),
                     xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
have_bleu = [s for s in lora_runs if s["name"] in results]
if have_bleu and full["name"] in results:
    axes[1].plot([s["trainable"] for s in have_bleu], [results[s["name"]]["bleu"] for s in have_bleu],
                 marker="o", label="LoRA")
    axes[1].axhline(results[full["name"]]["bleu"], color="C3", ls="--", label="full fine-tuning")
    axes[1].set_ylabel("BLEU on the E2E test set")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters")
    ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "knee.png"), dpi=150)
print("wrote knee.png")
