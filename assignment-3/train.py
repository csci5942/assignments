#!/usr/bin/env python3
"""Train assignment 1's `base` on one mix. GIVEN -- you do not edit this.

    python train.py --mix out/raw/mix_everything --name everything
    python train.py --mix out/raw/mix_english --name english --seed 2024

The model and every hyperparameter are assignment 1's `base`, unchanged.
The only thing that differs between your runs is the data your loader
feeds it -- which is the entire point of Part 3.

Writes out/<name>/ckpt.pt in assignment 1's checkpoint format, so
eval/eval.py can score it on the four corpora:

    python eval/eval.py --run out/<name> --name <name> --no-opik
"""

import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "data"))

from model import GPT, GPTConfig  # noqa: E402

# assignment 1's base config, verbatim
BASE = dict(n_layer=6, n_head=6, n_embd=384, dropout=0.2, block_size=256,
            batch_size=64, learning_rate=1e-3, lr_min=1e-4,
            weight_decay=0.1, warmup_iters=200)


def lr_at(step, iters, cfg):
    """Linear warmup, cosine decay to lr_min."""
    if step < cfg["warmup_iters"]:
        return cfg["learning_rate"] * (step + 1) / cfg["warmup_iters"]
    t = (step - cfg["warmup_iters"]) / max(1, iters - cfg["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return cfg["lr_min"] + coeff * (cfg["learning_rate"] - cfg["lr_min"])


def infinite(dl):
    """Loop the DataLoader forever; one pass is not enough steps."""
    while True:
        for batch in dl:
            yield batch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True, help="a mix directory of Parquet shards")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()

    # LOADER_MODULE=loader_solution runs the instructor version, which is
    # how the answer key is produced. Students leave it unset.
    import importlib
    _loader = importlib.import_module(os.environ.get("LOADER_MODULE", "loader"))
    MixDataset, load_vocab = _loader.MixDataset, _loader.load_vocab

    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfg = dict(BASE)
    stoi, itos = load_vocab(os.path.join(HERE, "data", "vocab.json"))
    ds = MixDataset(args.mix, block_size=cfg["block_size"], stoi=stoi,
                    seed=args.seed)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=cfg["batch_size"], num_workers=args.workers,
        pin_memory=(device == "cuda"), drop_last=True)

    mcfg = GPTConfig(block_size=cfg["block_size"], vocab_size=len(itos),
                     n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                     n_embd=cfg["n_embd"], dropout=cfg["dropout"])
    model = GPT(mcfg).to(device)
    n_params = model.num_params()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                            betas=(0.9, 0.99), weight_decay=cfg["weight_decay"])

    tok_per_iter = cfg["batch_size"] * cfg["block_size"]
    print(f"{args.name} (seed {args.seed}): {n_params:,} non-embedding params, "
          f"{len(ds.paths)} shards, {device}")
    print(f"  {args.iters} iters x {tok_per_iter:,} tokens = "
          f"{args.iters * tok_per_iter / 1e6:.1f}M tokens seen")

    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w") as f:
        f.write("step,tokens,train_loss,lr,seconds\n")

    batches = infinite(dl)
    t0, run = time.time(), 0.0
    for step in range(args.iters):
        lr = lr_at(step, args.iters, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = next(batches)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        run = 0.9 * run + 0.1 * loss.item() if step else loss.item()
        if step % 250 == 0 or step == args.iters - 1:
            secs = time.time() - t0
            print(f"  step {step:>6}  train {run:.4f}  lr {lr:.2e}  {secs:7.1f}s",
                  flush=True)
            with open(log_path, "a") as f:
                f.write(f"{step},{step*tok_per_iter},{run:.6f},{lr:.8f},{secs:.1f}\n")

    torch.save({"model": model.state_dict(), "config": mcfg.__dict__,
                "step": args.iters, "val_loss": run, "n_params": n_params},
               os.path.join(out_dir, "ckpt.pt"))
    summary = {"name": args.name, "seed": args.seed, "iters": args.iters,
               "mix": os.path.abspath(args.mix), "n_params": n_params,
               "final_train_loss": run,
               "tokens_seen": args.iters * tok_per_iter,
               "wall_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {out_dir}/ckpt.pt")
    print(f"  next: python eval/eval.py --run {out_dir} --name {args.name} --no-opik")
    return 0


if __name__ == "__main__":
    sys.exit(main())
