"""Training loop for the from-scratch transformer.

    python train.py --config configs/base.json
    python train.py --config configs/base.json --out-dir out/my-run

Logs a CSV row per eval (step, tokens, approx FLOPs, losses, lr, wall
time) so runs can be compared and scaling curves fitted afterwards.
The 6ND FLOP estimate is the one from the scaling-law literature:
C = 6 * N * D for a forward+backward pass over D tokens with N
non-embedding parameters.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))


def get_batch(split_data: np.ndarray, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(split_data[i: i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(split_data[i + 1: i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def amp_dtype_for(device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, amp_dtype, iters=40):
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def lr_at(step, cfg):
    """Linear warmup, cosine decay to lr_min."""
    if step < cfg["warmup_iters"]:
        return cfg["learning_rate"] * (step + 1) / cfg["warmup_iters"]
    t = (step - cfg["warmup_iters"]) / max(1, cfg["max_iters"] - cfg["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return cfg["lr_min"] + coeff * (cfg["learning_rate"] - cfg["lr_min"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    run_name = cfg.get("name") or os.path.splitext(os.path.basename(args.config))[0]
    out_dir = args.out_dir or os.path.join(HERE, "out", run_name)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    with open(os.path.join(HERE, "data", "meta.json")) as f:
        meta = json.load(f)
    train_data = np.fromfile(os.path.join(HERE, "data", "train.bin"), dtype=np.uint16)
    val_data = np.fromfile(os.path.join(HERE, "data", "val.bin"), dtype=np.uint16)

    mcfg = GPTConfig(
        block_size=cfg["block_size"],
        vocab_size=meta["vocab_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg.get("dropout", 0.2),
    )
    model = GPT(mcfg).to(device)
    n_params = model.num_params()
    amp_dtype = amp_dtype_for(device)
    print(f"run {run_name}: {n_params:,} non-embedding params, device {device}, "
          f"autocast {str(amp_dtype).replace('torch.', '')}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(0.9, 0.99),
        weight_decay=cfg.get("weight_decay", 0.1),
    )
    if cfg.get("compile", True) and device == "cuda":
        model = torch.compile(model)

    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w") as f:
        f.write("step,tokens,flops,train_loss,val_loss,lr,seconds\n")

    tokens_per_iter = cfg["batch_size"] * cfg["block_size"]
    best_val = float("inf")
    t0 = time.time()
    for step in range(cfg["max_iters"] + 1):
        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % cfg["eval_interval"] == 0 or step == cfg["max_iters"]:
            train_loss = estimate_loss(model, train_data, cfg["block_size"], cfg["batch_size"], device, amp_dtype)
            val_loss = estimate_loss(model, val_data, cfg["block_size"], cfg["batch_size"], device, amp_dtype)
            tokens = step * tokens_per_iter
            flops = 6 * n_params * tokens
            secs = time.time() - t0
            print(f"step {step:6d}  train {train_loss:.4f}  val {val_loss:.4f}  lr {lr:.2e}  {secs:7.1f}s")
            with open(log_path, "a") as f:
                f.write(f"{step},{tokens},{flops},{train_loss:.6f},{val_loss:.6f},{lr:.8f},{secs:.1f}\n")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model": model.state_dict(), "config": mcfg.__dict__, "step": step,
                     "val_loss": val_loss, "n_params": n_params},
                    os.path.join(out_dir, "ckpt.pt"),
                )
        if step == cfg["max_iters"]:
            break

        x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    summary = {"name": run_name, "n_params": n_params, "best_val_loss": best_val,
               "max_iters": cfg["max_iters"], "tokens": cfg["max_iters"] * tokens_per_iter,
               "flops": 6 * n_params * cfg["max_iters"] * tokens_per_iter}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
