"""Contiguous vs paged KV cache on one batch of prompts of mixed length.

    python bench_kv.py --model gpt2-xl --batch 32 --tokens 128
    python bench_kv.py --model gpt2-xl --batch 32 --tokens 128 --block-size 8 16 32 64 128 256

Prompts are random token ids with lengths drawn uniformly from
[--min-len, --max-len]. Every request generates --tokens tokens (greedy).
For each cache, prints prefill time, ms per decode step, and the KV
slots reserved vs holding a real token at the end of the run.
"""

import argparse
import time

import torch

from model import KVCache, from_pretrained, left_pad
from paged import PagedKVCache


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-xl")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--min-len", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--block-size", type=int, nargs="+", default=[16])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev, dtype = "cuda", torch.bfloat16
    model = from_pretrained(args.model).to(dev, dtype)
    cfg = model.config
    g = torch.Generator().manual_seed(args.seed)
    lens = torch.randint(args.min_len, args.max_len + 1, (args.batch,), generator=g)
    prompts = [torch.randint(0, cfg.vocab_size, (n,), generator=g).tolist() for n in lens]
    kv_bytes = 2 * cfg.n_layer * cfg.n_embd * 2          # per token, bf16
    print(f"{args.model}, batch {args.batch}, prompts {lens.min()}-{lens.max()} tokens "
          f"(mean {lens.float().mean():.0f}), {args.tokens} new tokens each, "
          f"KV {kv_bytes // 1024} KiB/token")
    print(f"{'cache':>16} {'prefill ms':>10} {'decode ms/step':>14} {'reserved':>9} {'used':>8} {'used %':>7}")

    def row(name, prefill, decode, stats):
        r, u = stats["reserved"], stats["used"]
        print(f"{name:>16} {prefill * 1e3:10.0f} {decode / (args.tokens - 1) * 1e3:14.1f} "
              f"{r:9d} {u:8d} {100 * u / r:6.1f}%")

    def decode(logits, cache):
        for _ in range(args.tokens - 1):
            logits, _ = model(logits[:, -1:].argmax(-1), cache=cache)

    with torch.no_grad():
        # Contiguous, left-padded: every row reserves T_max + tokens slots.
        idx, pad = left_pad(prompts)
        idx, pad = idx.to(dev), pad.to(dev)
        model.generate(idx[:, :8], 2, temperature=0, use_cache=True)   # warm up
        cache = KVCache(cfg, args.batch, idx.size(1) + args.tokens, pad=pad, device=dev, dtype=dtype)
        (logits, _), t_pre = timed(lambda: model(idx, cache=cache))
        _, t_dec = timed(lambda: decode(logits, cache))
        stats = cache.stats()
        stats["reserved"] = args.batch * (idx.size(1) + args.tokens)
        row("contiguous", t_pre, t_dec, stats)
        n = args.batch * cfg.block_size
        print(f"{'(max_len 1024)':>16} {'':>10} {'':>14} {n:9d} {stats['used']:8d} "
              f"{100 * stats['used'] / n:6.1f}%")
        del cache

        # Paged: prefill each prompt alone (no pad tokens), then decode together.
        for bs in args.block_size:
            num_blocks = sum(-(-(int(n) + args.tokens) // bs) for n in lens)
            cache = PagedKVCache(cfg, num_blocks, bs, device=dev, dtype=dtype)

            def prefill():
                last = []
                for s, p in enumerate(prompts):
                    cache.add(s)
                    cache.select([s])
                    last.append(model(torch.tensor([p], device=dev), cache=cache)[0][:, -1:])
                cache.select(range(args.batch))
                return torch.cat(last)
            logits, t_pre = timed(prefill)
            _, t_dec = timed(lambda: decode(logits, cache))
            row(f"paged, block {bs}", t_pre, t_dec, cache.stats())
            del cache


if __name__ == "__main__":
    main()
