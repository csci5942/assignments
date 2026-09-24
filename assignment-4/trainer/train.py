"""The training loop, on one GPU or on a mesh of them.

    python -m trainer.train --config configs/small.json --steps 50

At world size 1 this is exactly the single-GPU loop it has always been:
no process group is created and nothing distributed is touched. Under
`srun` it reads its rank from the environment, builds the mesh, and
wraps the model in whichever parallelism you asked for:

    srun python -m trainer.train --config configs/medium.json \
         --dp 4 --ddp overlapped

`--ddp naive|overlapped|torch` selects your Part 1 implementation, or
PyTorch's, which is the comparison Part 1 ends with.
"""

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

from .data import (Batcher, Prefetcher, ValBatcher, micro_batch_seed,
                   open_corpus, corpus_meta, DEFAULT_ROOT)
from .mesh import DEFAULT_ORDER, Mesh, parse_order
from .model import GPT, GPTConfig

# Dense bf16 tensor-core peak, TFLOP/s, for the MFU denominator. These
# are the vendor's numbers without the 2x sparsity asterisk.
PEAK_TFLOPS = {
    "A40": 149.7,
    "A100": 312.0,
    "H200": 989.0,
}


def peak_flops(device: str = "cuda") -> float:
    name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else ""
    for key, tf in PEAK_TFLOPS.items():
        if key.lower() in name.lower():
            return tf * 1e12
    return float("nan")


def lr_at(step: int, total: int, cfg: dict) -> float:
    """Linear warmup, cosine decay to lr_min. A1's schedule, unchanged."""
    warmup = cfg["warmup_steps"]
    if step < warmup:
        return cfg["lr"] * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return cfg["lr_min"] + coeff * (cfg["lr"] - cfg["lr_min"])


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    defaults = dict(micro_batch=4, global_batch=128, lr=6e-4, lr_min=6e-5,
                    warmup_steps=20, weight_decay=0.1, grad_clip=1.0,
                    beta1=0.9, beta2=0.95, seed=1337)
    return {**defaults, **cfg}


@torch.no_grad()
def evaluate(model, val_batcher, device) -> float:
    """Mean loss over the fixed validation blocks."""
    model.eval()
    losses = []
    for x, y in val_batcher:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=str(device).startswith("cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def memory_report(model, device) -> dict:
    if not str(device).startswith("cuda"):
        return {}
    buckets = model.param_bytes()
    peak = torch.cuda.max_memory_allocated()
    known = sum(buckets.values())
    return {**buckets,
            "measured_peak": peak,
            "activations_residual": peak - known}


def mean_over_dp(value: float, mesh: Mesh, device) -> float:
    """Average a scalar across the data parallel group, for reporting.

    Every rank saw a different quarter of the batch, so every rank has a
    different loss. The number worth printing is the one the global batch
    would have produced, which is their mean -- and it costs one tiny
    all-reduce per step, off the critical path of the gradient.
    """
    if mesh.dp <= 1 or not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=mesh.dp_group)
    return (t / mesh.dp).item()


def clip_grad_norm(params, max_norm: float, mesh: Mesh) -> float:
    """Clip by the gradient norm of the *whole model*, not this rank's part.

    `torch.nn.utils.clip_grad_norm_` sums over the parameters it is
    given, which under tensor or pipeline parallelism is only a slice of
    the model. Every rank then computes a different norm, clips by a
    different factor, and the run quietly stops being the computation a
    single GPU would have performed. Measured on `small` at tp=4: the
    dense norm was 1.382 and a rank's local norm 0.924, so with a clip
    threshold of 1.0 the dense model clipped and the tensor parallel one
    did not.

    Three kinds of parameter, three treatments:

    * **Tensor-parallel shards** (tagged in `tp.py`) are disjoint across
      the tp group, so their squares are summed across it.
    * **Replicated** parameters -- embeddings, layer norms -- are
      identical on every tp rank, so they are counted once and not
      reduced.
    * **Pipeline stages** hold disjoint parameters, so the total is
      summed across the pp group.

    Data parallel needs nothing: gradients are already averaged, so every
    replica holds the same ones.
    """
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    device = params[0].grad.device
    shard = torch.zeros((), dtype=torch.float32, device=device)
    repl = torch.zeros((), dtype=torch.float32, device=device)
    for p in params:
        sq = p.grad.detach().float().pow(2).sum()
        if getattr(p, "tensor_parallel", False):
            shard += sq
        else:
            repl += sq

    if mesh.tp > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(shard, op=dist.ReduceOp.SUM, group=mesh.tp_group)
    total = shard + repl
    if mesh.pp > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=mesh.pp_group)

    norm = total.sqrt()
    if max_norm and max_norm > 0:
        scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        for p in params:
            p.grad.detach().mul_(scale.to(p.grad.dtype))
    return norm.item()


def wrap_model(model, mesh: Mesh, strategy: str):
    """Return (model_to_run, finish_fn, no_sync_fn) for the chosen strategy.

    `finish_fn()` must leave every gradient reduced and safe to read;
    `no_sync_fn(is_last)` gives back a context manager for a micro-batch.
    Keeping both behind this interface is what lets the loop below stay
    the same shape for all four strategies.
    """
    if mesh.dp <= 1 or strategy == "none":
        return model, (lambda: None), (lambda last: contextlib.nullcontext())

    if strategy == "torch":
        from torch.nn.parallel import DistributedDataParallel
        device_ids = [mesh.local_rank] if torch.cuda.is_available() else None
        ddp = DistributedDataParallel(model, device_ids=device_ids,
                                      process_group=mesh.dp_group)
        return (ddp, (lambda: None),
                lambda last: contextlib.nullcontext() if last else ddp.no_sync())

    if strategy == "naive":
        from .ddp import broadcast_module_state, naive_all_reduce_grads
        broadcast_module_state(model, group=mesh.dp_group)
        return (model,
                lambda: naive_all_reduce_grads(model, group=mesh.dp_group),
                # The naive version reduces once, after everything.
                lambda last: contextlib.nullcontext())

    if strategy == "overlapped":
        from .ddp import OverlappedDDP
        ddp = OverlappedDDP(model, group=mesh.dp_group)
        return (ddp, ddp.finish_gradient_synchronization,
                lambda last: contextlib.nullcontext() if last else ddp.no_sync())

    raise ValueError(f"unknown --ddp strategy {strategy!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=DEFAULT_ROOT)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup-steps", type=int, default=5,
                    help="excluded from the timing average, not the LR warmup")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="overrides the config; the knob that trades memory for throughput")
    ap.add_argument("--global-batch", type=int, default=None,
                    help="overrides the config. Fixed at 128 everywhere "
                         "except Part 3, where the last pipeline stage's "
                         "logits make the full batch impossible under AFAB")
    ap.add_argument("--eval-every", type=int, default=0, help="0 disables")
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--out", default=None, help="directory for log.csv and summary.json")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output directory, so a sweep does "
                         "not overwrite itself")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--dp", type=int, default=None,
                    help="data parallel degree; defaults to the whole world")
    ap.add_argument("--tp", type=int, default=1, help="Part 2")
    ap.add_argument("--pp", type=int, default=1, help="Part 3")
    ap.add_argument("--axis-order", default=",".join(DEFAULT_ORDER),
                    help="which dimension crosses the node boundary; Part 4")
    ap.add_argument("--schedule", default="1f1b", choices=("afab", "1f1b"),
                    help="pipeline schedule; Part 3")
    ap.add_argument("--ddp", default="overlapped",
                    choices=("none", "naive", "overlapped", "torch"),
                    help="'torch' is PyTorch's DistributedDataParallel, the "
                         "comparison Part 1 ends with")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.micro_batch is not None:
        cfg["micro_batch"] = args.micro_batch
    if args.global_batch is not None:
        cfg["global_batch"] = args.global_batch
    if args.seed is not None:
        cfg["seed"] = args.seed
    name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])

    # ---- mesh ------------------------------------------------------
    # At world size 1 this creates no process group and the run is the
    # plain single-GPU one.
    order = parse_order(args.axis_order)
    world = int(os.environ.get("WORLD_SIZE")
                or os.environ.get("SLURM_NTASKS") or 1)
    dp = args.dp if args.dp is not None else world // (args.tp * args.pp)
    if dp * args.tp * args.pp != world:
        print(f"dp*tp*pp = {dp}*{args.tp}*{args.pp} = "
              f"{dp * args.tp * args.pp}, but the world has {world} ranks",
              file=sys.stderr)
        return 2
    mesh = Mesh.from_env(dp=dp, tp=args.tp, pp=args.pp, order=order)
    lead = mesh.is_leader

    def say(*a, **kw):
        if lead:
            print(*a, **kw)

    device = mesh.device
    # Every rank draws the same weights. `broadcast_module_state` makes
    # that guaranteed rather than incidental, but seeding identically
    # means the broadcast has nothing to correct in the common case.
    torch.manual_seed(cfg["seed"])
    if device.startswith("cuda"):
        torch.cuda.set_device(mesh.local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()

    # ---- data ------------------------------------------------------
    mcfg = GPTConfig.from_dict(cfg)
    shards = open_corpus(args.data_root, mcfg.block_size, "train")
    batcher = Batcher(shards, cfg["global_batch"], seed=cfg["seed"])
    val = None
    if args.eval_every:
        val_shards = open_corpus(args.data_root, mcfg.block_size, "val")
        val = ValBatcher(val_shards, cfg["micro_batch"], args.eval_batches)

    # ---- model -----------------------------------------------------
    # Build dense, then shard. `parallelize_gpt` takes each rank's slice
    # of the weights the dense model already drew, so every rank's shard
    # comes from the same initialization -- which is what makes a tensor
    # parallel run and a single-GPU run the same computation.
    model = GPT(mcfg).to(device)
    # Accounting is taken from the dense model, before sharding. FLOPs
    # per token is a property of the *model*, not of how it is split:
    # the mesh still does all of that arithmetic, just spread out. Taking
    # it after sharding would divide it by tp and inflate MFU to match.
    dense_fpt = model.flops_per_token()
    dense_params = model.num_params()
    dense_non_embedding = model.num_params(non_embedding=True)
    if mesh.tp > 1:
        from .tp import TensorParallelGPT
        model = TensorParallelGPT(model, group=mesh.tp_group).to(device)
    stage = None
    if mesh.pp > 1:
        from .pp import PipelineStage
        # The stage holds a slice of the very model just built, so a
        # pipelined run and a single-GPU run start from identical
        # weights. Only this rank's slice stays on the device.
        stage = PipelineStage(model, mesh).to(device)
        model = stage
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            betas=(cfg["beta1"], cfg["beta2"]),
                            weight_decay=cfg["weight_decay"],
                            fused=device.startswith("cuda"))
    run_model, finish_grads, micro_ctx = wrap_model(model, mesh, args.ddp)

    micro = cfg["micro_batch"]
    # The accumulation count is over this rank's *slice* of the global
    # batch, not the global batch. Getting this wrong is the quiet way to
    # train on dp times too much data.
    if cfg["global_batch"] % mesh.dp:
        print(f"global_batch {cfg['global_batch']} is not divisible by "
              f"dp {mesh.dp}", file=sys.stderr)
        return 2
    local_batch = cfg["global_batch"] // mesh.dp
    if local_batch % micro:
        print(f"this rank's {local_batch} sequences "
              f"(global_batch {cfg['global_batch']} / dp {mesh.dp}) are not "
              f"divisible by micro_batch {micro}", file=sys.stderr)
        return 2
    accum = local_batch // micro
    if args.steps <= args.warmup_steps:
        # Every step would be excluded from the timing and the reported
        # throughput would be a nan. Time everything and say so, rather
        # than printing a number that is not one.
        if lead:
            print(f"  note: --steps {args.steps} is not more than "
                  f"--warmup-steps {args.warmup_steps}, so no step would be "
                  f"timed. Timing all of them; treat the throughput as "
                  f"indicative only.", file=sys.stderr)
        args.warmup_steps = 0
    fpt = dense_fpt
    tok_per_step = batcher.tokens_per_step

    say(f"{name}: {dense_params:,} params "
        f"({dense_non_embedding:,} non-embedding)")
    say(f"  {shards}")
    say(f"  device {torch.cuda.get_device_name() if device.startswith('cuda') else 'cpu'}"
        f" x {mesh.world_size}")
    if mesh.world_size > 1:
        say(f"  mesh dp={mesh.dp} tp={mesh.tp} pp={mesh.pp}, "
            f"order {'/'.join(mesh.order)}, --ddp {args.ddp}")
    say(f"  global batch {cfg['global_batch']} x {mcfg.block_size} = "
        f"{tok_per_step:,} tokens/step, {local_batch} per rank, "
        f"micro-batch {micro}, "
        f"{accum} accumulation step{'s' if accum > 1 else ''}")
    say(f"  {args.steps} steps = {args.steps * tok_per_step / 1e6:.1f}M tokens")

    out_dir = args.out or os.path.join("out", name + (f"-{args.tag}" if args.tag else ""))
    # Only the leader writes. Every rank writing the same path is a race
    # that produces one interleaved, unreadable log.csv.
    prior = os.path.join(out_dir, "summary.json")
    if lead and os.path.exists(prior):
        # A micro-batch sweep, and Part 4's sweep over everything,
        # both land here. Without a --tag those runs all land here and quietly
        # replace each other, and nothing in the file that survives says
        # which settings produced it.
        try:
            with open(prior) as f:
                was = json.load(f)
            now = {"micro_batch": micro, "world_size": mesh.world_size,
                   "steps": args.steps, "config": args.config,
                   "strategy": args.ddp}
            changed = [k for k in now if k in was and was[k] != now[k]]
            if changed:
                print(f"  warning: overwriting {prior}, which was written "
                      f"with a different {', '.join(changed)}. "
                      f"Use --tag to keep both.", file=sys.stderr)
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    log_path = os.path.join(out_dir, "log.csv")
    if lead:
        os.makedirs(out_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write("step,tokens,loss,lr,step_seconds,tokens_per_second,mfu\n")

    # ---- loop ------------------------------------------------------
    # MFU's denominator is the arithmetic the *whole mesh* could have
    # done, so it scales with the number of GPUs. Dividing the global
    # throughput by one GPU's peak is how you get a 126% MFU on four
    # A40s, which is a good sign you divided by the wrong thing.
    peak_fl = peak_flops(device) * mesh.world_size
    step_times, smoothed = [], None
    busy_seconds = 0.0
    step_compute = 0.0
    cuda = device.startswith("cuda")
    stochastic = mcfg.dropout > 0.0
    wall0 = time.time()
    prefetch = Prefetcher(batcher, start=0, depth=2,
                          dp_rank=mesh.dp_rank, dp_world=mesh.dp)

    mesh.barrier()          # start the clock together, not as ranks arrive
    for step in range(args.steps):
        lr = lr_at(step, args.steps, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        if cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        x_all, y_all = prefetch.next()
        opt.zero_grad(set_to_none=True)
        total = 0.0

        if stage is not None:
            # Pipeline parallel replaces the accumulation loop entirely:
            # the schedule decides the order, and `busy` is the time this
            # stage actually spent computing, which is what the bubble
            # deliverable compares against the step time.
            from .pp import pipeline_step
            micro_batches = [
                (x_all[i * micro:(i + 1) * micro].to(device),
                 y_all[i * micro:(i + 1) * micro].to(device))
                for i in range(accum)]
            total, compute = pipeline_step(
                stage, mesh, micro_batches, schedule=args.schedule,
                device=device,
                dtype=(torch.bfloat16 if cuda else torch.float32),
                scale=accum)
            total = total or 0.0
            step_compute = compute
        else:
          for i in range(accum):
              if stochastic:
                  # Key the dropout masks to this micro-batch's *global*
                  # index, so the run is the same computation at any dp
                  # degree. See data.micro_batch_seed.
                  torch.manual_seed(
                      micro_batch_seed(cfg["seed"], step,
                                       mesh.dp_rank * accum + i))
              x = x_all[i * micro:(i + 1) * micro].to(device, non_blocking=True)
              y = y_all[i * micro:(i + 1) * micro].to(device, non_blocking=True)
              with micro_ctx(i == accum - 1):
                  with torch.autocast(device_type="cuda",
                                      dtype=torch.bfloat16, enabled=cuda):
                      _, loss = run_model(x, y)
                  # Each micro-batch contributes 1/accum of this rank's
                  # gradient; the data parallel average does the rest.
                  (loss / accum).backward()
              total += loss.item() / accum
        finish_grads()
        if cuda:
            # Let the gradient reduction finish before anything reads a
            # gradient. The reduction is hundreds of NCCL collectives
            # issued back to back, and NCCL returns from each as soon as
            # it is enqueued. Ranks do not arrive together -- across two
            # nodes we measured a two second spread -- and the next
            # operation to touch `p.grad` can then sit waiting for work
            # on peers that have not caught up, with no error and no
            # progress. One synchronisation a step costs nothing next to
            # the reduction itself and makes the step deterministic.
            torch.cuda.synchronize()
        clip_grad_norm(list(model.parameters()), cfg["grad_clip"], mesh)
        opt.step()

        if cuda:
            torch.cuda.synchronize()
        dt = time.time() - t0
        if step >= args.warmup_steps:
            step_times.append(dt)
            # Accumulated here, not in the branch above, so that busy and
            # step time cover the same steps. Counting busy over every
            # step while dividing by post-warmup steps only is how a
            # "busy fraction" ends up greater than one.
            busy_seconds += step_compute

        total = mean_over_dp(total, mesh, device)
        smoothed = total if smoothed is None else 0.9 * smoothed + 0.1 * total
        tps = tok_per_step / dt
        mfu = fpt * tok_per_step / dt / peak_fl if peak_fl == peak_fl else float("nan")

        if lead and (step % 10 == 0 or step == args.steps - 1):
            print(f"  step {step:>5}  loss {smoothed:6.3f}  lr {lr:.2e}  "
                  f"{dt * 1e3:7.1f} ms  {tps:9,.0f} tok/s  mfu {mfu * 100:5.1f}%",
                  flush=True)
        if lead:
            with open(log_path, "a") as f:
                f.write(f"{step},{(step + 1) * tok_per_step},{total:.6f},"
                        f"{lr:.8f},{dt:.6f},{tps:.1f},{mfu:.6f}\n")

        if val is not None and args.eval_every and \
                (step + 1) % args.eval_every == 0:
            v = mean_over_dp(evaluate(model, val, device), mesh, device)
            say(f"  step {step:>5}  val {v:.4f}", flush=True)

    prefetch.close()

    # ---- summary ---------------------------------------------------
    wall = time.time() - wall0
    median = statistics.median(step_times) if step_times else float("nan")
    steady_tps = tok_per_step / median
    summary = {
        "name": name,
        "config": args.config,
        "params": dense_params,
        "params_per_rank": sum(p.numel() for p in model.parameters()),
        "device": (torch.cuda.get_device_name() if cuda else "cpu"),
        "world_size": mesh.world_size,
        "dp": mesh.dp,
        "tp": mesh.tp,
        "pp": mesh.pp,
        "axis_order": "/".join(mesh.order),
        "strategy": args.ddp,
        "steps": args.steps,
        "global_batch": cfg["global_batch"],
        "micro_batch": micro,
        "accum": accum,
        "block_size": mcfg.block_size,
        "tokens": args.steps * tok_per_step,
        "final_loss": smoothed,
        "median_step_seconds": median,
        "tokens_per_second": steady_tps,
        "mfu": fpt * steady_tps / peak_fl if peak_fl == peak_fl else None,
        "wall_seconds": round(wall, 1),
        "schedule": args.schedule if mesh.pp > 1 else None,
        "busy_fraction": (busy_seconds / sum(step_times)
                          if stage is not None and step_times
                          else None),
        "memory": memory_report(model, device),
    }
    if val is not None:
        summary["val_loss"] = mean_over_dp(evaluate(model, val, device),
                                           mesh, device)

    if lead:
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  steady state: {median * 1e3:.1f} ms/step, "
              f"{steady_tps:,.0f} tok/s"
              + (f", {summary['mfu'] * 100:.1f}% MFU" if summary["mfu"] else ""))
        if summary["memory"]:
            m = summary["memory"]
            print(f"  memory: {m['measured_peak'] / 2**30:.2f} GiB peak, "
                  f"per rank "
                  f"({(m['params'] + m['grads'] + m['optimizer']) / 2**30:.2f} "
                  f"predicted for params+grads+optimizer, "
                  f"{m['activations_residual'] / 2**30:.2f} unexplained)")
        print(f"  -> {out_dir}/summary.json")
    Mesh.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
