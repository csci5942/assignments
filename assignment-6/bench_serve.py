"""Static vs continuous batching on one offline workload.

    python bench_serve.py --engine static continuous
    python bench_serve.py --engine continuous --optimistic     # needs preempt()

All requests arrive at t=0. Prompts are random token ids, lengths uniform
in [--min-prompt, --max-prompt]; each request generates exactly max_new
tokens, uniform in [--min-new, --max-new] (EOS ignored). The KV pool is
--kv-gb of blocks. Prints throughput, request latency (time from the
start to the request's last token), mean decode batch, and preemptions.
"""

import argparse
import time

import torch

from engine import Engine, Request, Scheduler, StaticScheduler
from model import from_pretrained
from paged import PagedKVCache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-xl")
    ap.add_argument("--engine", nargs="+", default=["static", "continuous"])
    ap.add_argument("--optimistic", action="store_true", help="optimistic admission + preemption")
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--min-prompt", type=int, default=16)
    ap.add_argument("--max-prompt", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--kv-gb", type=float, default=4.0)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev, dtype = "cuda", torch.bfloat16
    model = from_pretrained(args.model).to(dev, dtype)
    cfg = model.config
    g = torch.Generator().manual_seed(args.seed)
    plens = torch.randint(args.min_prompt, args.max_prompt + 1, (args.requests,), generator=g).tolist()
    news = torch.randint(args.min_new, args.max_new + 1, (args.requests,), generator=g).tolist()
    prompts = [torch.randint(0, cfg.vocab_size, (n,), generator=g).tolist() for n in plens]
    kv_per_token = 2 * cfg.n_layer * cfg.n_embd * 2
    num_blocks = int(args.kv_gb * 1e9 // (kv_per_token * args.block_size))
    print(f"{args.model}: {args.requests} requests, prompts {args.min_prompt}-{args.max_prompt}, "
          f"outputs {args.min_new}-{args.max_new} (sum {sum(news)}), KV {args.kv_gb} GB = "
          f"{num_blocks} blocks of {args.block_size}, max batch {args.max_batch}")
    print(f"{'engine':>24} {'time s':>7} {'tok/s':>7} {'mean lat s':>10} {'p90 lat s':>9} "
          f"{'mean batch':>10} {'preempt':>7}")

    model.generate(torch.zeros(1, 8, dtype=torch.long, device=dev), 2, temperature=0, use_cache=True)
    for name in args.engine:
        cache = PagedKVCache(cfg, num_blocks, args.block_size, device=dev, dtype=dtype)
        cls = StaticScheduler if name == "static" else Scheduler
        sched = cls(cache, args.max_batch, optimistic=args.optimistic and name != "static")
        engine = Engine(model, cache, sched)
        reqs = [Request(i, p, n) for i, (p, n) in enumerate(zip(prompts, news))]
        t = time.perf_counter()
        engine.run(reqs)
        secs = time.perf_counter() - t
        lat = sorted(r.t_done for r in reqs)
        label = name + (" (optimistic)" if sched.optimistic else "")
        print(f"{label:>24} {secs:7.1f} {sum(news) / secs:7.0f} {sum(lat) / len(lat):10.1f} "
              f"{lat[int(0.9 * len(lat))]:9.1f} "
              f"{sum(engine.batch_sizes) / len(engine.batch_sizes):10.1f} "
              f"{sum(r.preemptions for r in reqs):7d}")
        del cache, engine


if __name__ == "__main__":
    main()
