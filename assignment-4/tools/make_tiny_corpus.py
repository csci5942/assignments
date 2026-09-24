#!/usr/bin/env python3
"""Build a throwaway corpus so you can run things before you have data.

    python tools/make_tiny_corpus.py --out /tmp/tiny

The tokens are random integers. Nothing trained on this will learn
anything, and the loss will sit at ln(vocab_size) forever -- that is
fine and expected. What it is for is checking that the plumbing works:
that a step runs, that throughput and memory get reported, that your
collectives agree with a single rank. Correctness does not care whether
the tokens mean anything.

Use it when your Delta allocation has not come through yet, when you are
working on a laptop, or when you want a fast loop that does not touch
/work/nvme.

For the real thing, see tools/prepare_fineweb.py -- though on Delta the
course corpus is already built and you should just use it.
"""

import argparse
import json
import os
import sys

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--tokens-per-shard", type=int, default=2_000_000)
    ap.add_argument("--val-tokens", type=int, default=500_000)
    ap.add_argument("--vocab-size", type=int, default=50257)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.vocab_size > 65536:
        print("vocab_size must fit in uint16", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.shards):
        rng.integers(0, args.vocab_size, size=args.tokens_per_shard,
                     dtype=np.uint16).tofile(
                         os.path.join(args.out, f"train_{i:04d}.bin"))
    rng.integers(0, args.vocab_size, size=args.val_tokens,
                 dtype=np.uint16).tofile(os.path.join(args.out, "val.bin"))

    total = args.shards * args.tokens_per_shard
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"tokenizer": "SYNTHETIC -- random ids, not text",
                   "vocab_size": args.vocab_size, "dtype": "uint16",
                   "train_tokens": total, "val_tokens": args.val_tokens,
                   "seed": args.seed}, f, indent=2)

    print(f"{total:,} train + {args.val_tokens:,} val tokens -> {args.out}")
    print(f"  python -m trainer.train --config configs/smoke.json "
          f"--data-root {args.out} --steps 20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
