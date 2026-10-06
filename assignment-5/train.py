"""Fine-tune GPT-2 on E2E, two ways: every weight, or LoRA adapters.

    python train.py --config configs/gpt2-medium-full.json
    python train.py --config configs/lora-r4.json --steps 50     # a smoke run

A4's single-GPU training loop with three changes: batches come from
sft.collate (Part 2), the model comes from model.from_pretrained, and
`mode: lora` applies lora.apply_lora (Part 4) before the optimizer is
built. The memory report at the end is A4's: the three buckets, the
measured peak, the residual. Not edited in Assignment 5.

Writes out/<name>/log.csv (step, loss, lr, seconds), summary.json, and a
checkpoint: ckpt.pt with the full state dict for `full`, lora.pt with the
adapters alone for `lora`.
"""

import argparse
import json
import math
import os
import time

import torch

import lora
import sft
from model import from_pretrained

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_EVERY = 20
DEV_EXAMPLES = 512          # how many dev pairs the end-of-run loss uses
DEV_BATCH = 32


def lr_at(step: int, total: int, lr: float, warmup: int) -> float:
    """Linear warmup, then cosine down to a tenth of lr."""
    if step < warmup:
        return lr * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return 0.1 * lr + 0.9 * lr * 0.5 * (1.0 + math.cos(math.pi * frac))


def memory_buckets(model) -> dict:
    """Bytes for weights, gradients and Adam state from the parameter
    counts: fp32 weights (the master copy under autocast), and a gradient
    plus two moments for trainable parameters only."""
    n_all = sum(p.numel() for p in model.parameters())
    n_train = lora.count_trainable(model)
    return {"params": 4 * n_all, "grads": 4 * n_train, "optimizer": 8 * n_train}


@torch.no_grad()
def dev_loss(model, examples, device) -> float:
    """Mean masked loss over the first DEV_EXAMPLES dev pairs."""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(examples), DEV_BATCH):
        x, y = sft.collate(examples[i:i + DEV_BATCH])
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            _, loss = model(x, y)
        total += loss.item()
        n += 1
    model.train()
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="default out/<name>")
    ap.add_argument("--steps", type=int, default=None, help="override the epoch count")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    out_dir = args.out or os.path.join(HERE, "out", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    device = args.device
    torch.manual_seed(cfg["seed"])

    # ---- data: every pair formatted once, up front
    train_rows = sft.load_split("train")
    dev_rows = sft.load_split("dev")[:DEV_EXAMPLES]
    train = [sft.format_example(r["mr"], r["ref"]) for r in train_rows]
    dev = [sft.format_example(r["mr"], r["ref"]) for r in dev_rows]
    steps_per_epoch = math.ceil(len(train) / cfg["batch"])
    total_steps = args.steps or cfg["epochs"] * steps_per_epoch

    # ---- model: pretrained GPT-2, then either everything trains or LoRA
    model = from_pretrained(cfg["model"])
    if cfg["mode"] == "lora":
        trainable = lora.apply_lora(model, cfg["rank"], cfg["alpha"],
                                    tuple(cfg["targets"]))
    elif cfg["mode"] == "full":
        for p in model.parameters():
            p.requires_grad = True
        trainable = lora.count_trainable(model)
    else:
        raise ValueError(f"mode must be full or lora, got {cfg['mode']}")
    buckets = memory_buckets(model)
    model.to(device)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"{cfg['name']}: {cfg['model']}, mode {cfg['mode']}"
          + (f" r={cfg['rank']} alpha={cfg['alpha']} on {cfg['targets']}" if cfg["mode"] == "lora" else "")
          + f"\n  {trainable:,} trainable of {n_all:,} ({100 * trainable / n_all:.2f}%)"
          f"\n  {len(train):,} pairs, batch {cfg['batch']}, {total_steps} steps"
          f" ({total_steps / steps_per_epoch:.2f} epochs), lr {cfg['lr']}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.95),
                            weight_decay=cfg["weight_decay"])

    # ---- the loop
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    g = torch.Generator().manual_seed(cfg["seed"])
    order = torch.randperm(len(train), generator=g).tolist()
    cursor = 0
    log = open(os.path.join(out_dir, "log.csv"), "w")
    log.write("step,loss,lr,seconds\n")
    losses = []
    tokens_seen = 0
    model.train()
    t0 = time.time()
    for step in range(total_steps):
        if cursor + cfg["batch"] > len(order):          # new epoch, reshuffle
            order = torch.randperm(len(train), generator=g).tolist()
            cursor = 0
        batch = [train[i] for i in order[cursor:cursor + cfg["batch"]]]
        cursor += cfg["batch"]
        x, y = sft.collate(batch)
        x, y = x.to(device), y.to(device)
        tokens_seen += x.numel()

        lr = lr_at(step, total_steps, cfg["lr"], cfg["warmup_steps"])
        for group in opt.param_groups:
            group["lr"] = lr
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
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
            print(f"  step {step:5d}/{total_steps}  loss {loss.item():.4f}  "
                  f"lr {lr:.2e}  {time.time() - t0:6.0f}s")
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - t0
    log.close()

    # ---- the numbers the assignment is about
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    known = sum(buckets.values())
    memory = {**buckets, "measured_peak": peak, "activations_residual": peak - known}
    final = sum(losses[-50:]) / len(losses[-50:])
    dloss = dev_loss(model, dev, device)
    summary = {
        "name": os.path.basename(os.path.normpath(out_dir)), "config": cfg,
        "params": n_all, "trainable": trainable,
        "steps": total_steps, "epochs": total_steps / steps_per_epoch,
        "tokens_seen": tokens_seen,
        "seconds": seconds, "steps_per_second": total_steps / seconds,
        "train_loss_last50": final, "dev_loss": dloss,
        "memory": memory,
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- checkpoint: the whole model for full, the adapters alone for lora
    if cfg["mode"] == "lora":
        torch.save({"lora": lora.lora_state_dict(model), "config": cfg},
                   os.path.join(out_dir, "lora.pt"))
    else:
        torch.save({"model": model.state_dict(), "config": cfg},
                   os.path.join(out_dir, "ckpt.pt"))

    gib = 2 ** 30
    print(f"\n  done: {seconds:.0f}s, train loss {final:.3f}, dev loss {dloss:.3f}")
    print(f"  memory: params {buckets['params'] / gib:.2f} GiB, grads "
          f"{buckets['grads'] / gib:.2f}, optimizer {buckets['optimizer'] / gib:.2f}, "
          f"measured peak {peak / gib:.2f}, residual {(peak - known) / gib:.2f}")
    print(f"  -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
