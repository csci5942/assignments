"""Rebuild the starting policy: A5's Part 3, one epoch of full fine-tuning
of GPT-2 medium on E2E, in case your A5 checkpoint is gone.

    python train_sft.py                       # writes out/sft, about 3 minutes on a 4090

A5's train.py with the LoRA path and the memory report removed. If you
still have A5's run, skip this and link it instead:

    ln -s ../../assignment-5/out/gpt2-medium-full out/sft

Not edited in Assignment 7.
"""

import json
import math
import os
import time

import torch

import sft
from model import from_pretrained

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "configs", "sft.json")
OUT = os.path.join(HERE, "out", "sft")
LOG_EVERY = 20
DEV_EXAMPLES = 512
DEV_BATCH = 32


def lr_at(step, total, lr, warmup):
    """Linear warmup, then cosine down to a tenth of lr."""
    if step < warmup:
        return lr * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return 0.1 * lr + 0.9 * lr * 0.5 * (1.0 + math.cos(math.pi * frac))


@torch.no_grad()
def dev_loss(model, examples, device):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(examples), DEV_BATCH):
        x, y = sft.collate(examples[i:i + DEV_BATCH])
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)
        total += loss.item()
        n += 1
    model.train()
    return total / n


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(CONFIG) as f:
        cfg = json.load(f)
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(cfg["seed"])

    train_rows = sft.load_split("train")
    dev_rows = sft.load_split("dev")[:DEV_EXAMPLES]
    train = [sft.format_example(r["mr"], r["ref"]) for r in train_rows]
    dev = [sft.format_example(r["mr"], r["ref"]) for r in dev_rows]
    steps_per_epoch = math.ceil(len(train) / cfg["batch"])
    total_steps = cfg["epochs"] * steps_per_epoch

    model = from_pretrained(cfg["model"]).to(device)
    params = [p for p in model.parameters()]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg["weight_decay"])
    print(f"sft: {cfg['model']}, {len(train):,} pairs, batch {cfg['batch']}, {total_steps} steps, lr {cfg['lr']}")

    g = torch.Generator().manual_seed(cfg["seed"])
    order = torch.randperm(len(train), generator=g).tolist()
    cursor = 0
    log = open(os.path.join(OUT, "log.csv"), "w")
    log.write("step,loss,lr,seconds\n")
    losses = []
    model.train()
    t0 = time.time()
    for step in range(total_steps):
        if cursor + cfg["batch"] > len(order):
            order = torch.randperm(len(train), generator=g).tolist()
            cursor = 0
        batch = [train[i] for i in order[cursor:cursor + cfg["batch"]]]
        cursor += cfg["batch"]
        x, y = sft.collate(batch)
        x, y = x.to(device), y.to(device)
        lr = lr_at(step, total_steps, cfg["lr"], cfg["warmup_steps"])
        for group in opt.param_groups:
            group["lr"] = lr
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()
        losses.append(loss.item())
        if step % LOG_EVERY == 0 or step == total_steps - 1:
            log.write(f"{step},{loss.item():.4f},{lr:.3e},{time.time() - t0:.1f}\n")
            log.flush()
        if step % 200 == 0 or step == total_steps - 1:
            print(f"  step {step:5d}/{total_steps}  loss {loss.item():.4f}  lr {lr:.2e}  {time.time() - t0:6.0f}s")
    log.close()
    seconds = time.time() - t0

    summary = {"name": "sft", "config": cfg, "steps": total_steps, "seconds": seconds,
               "train_loss_last50": sum(losses[-50:]) / len(losses[-50:]),
               "dev_loss": dev_loss(model, dev, device),
               "device": torch.cuda.get_device_name() if device == "cuda" else device}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(OUT, "ckpt.pt"))
    print(f"\n  done: {seconds:.0f}s, train loss {summary['train_loss_last50']:.3f}, "
          f"dev loss {summary['dev_loss']:.3f}\n  -> {OUT}/summary.json, ckpt.pt")


if __name__ == "__main__":
    main()
