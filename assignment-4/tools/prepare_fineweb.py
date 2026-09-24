#!/usr/bin/env python3
"""Tokenize FineWeb-Edu into the .bin shards trainer/data.py reads.

**You do not need to run this.** The course corpus is already built and
staged on Delta at /work/nvme/bidb/csci5942/a4/fineweb. This script is
here so you can see exactly what those files are, and so you can rebuild
them somewhere else if you ever want to.

    # download the parquet (needs the internet)
    python tools/prepare_fineweb.py --stage download --raw $RAW

    # tokenize it (needs cores, not the internet)
    python tools/prepare_fineweb.py --stage tokenize --raw $RAW --out $OUT

    # or both at once
    python tools/prepare_fineweb.py --stage all --raw $RAW --out $OUT

The two stages are separate because on most clusters the machine with
the network and the machine with the cores are not the same machine.
Both are resumable: `download` skips files it already has, `tokenize`
skips shards it has already written, so a job that hits its wall clock
can simply be resubmitted.

Output is raw little-endian uint16 token ids, no header -- GPT-2's 50257
tokens fit in 16 bits, so the corpus is two bytes per token and
`np.memmap` addresses it directly. Alongside the shards, `meta.json`
records what produced them, because a corpus with no provenance is not a
corpus, it is a pile of bytes.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ID = "HuggingFaceFW/fineweb-edu"
SAMPLE = "sample/10BT"


def human(n: int) -> str:
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


# ---------------------------------------------------------------------
# stage 1: download
# ---------------------------------------------------------------------
def download(raw_dir: str, sample: str, max_files: int) -> list:
    from huggingface_hub import hf_hub_download, list_repo_files

    os.makedirs(raw_dir, exist_ok=True)
    names = sorted(f for f in list_repo_files(REPO_ID, repo_type="dataset")
                   if f.startswith(sample) and f.endswith(".parquet"))
    if max_files:
        names = names[:max_files]
    print(f"{len(names)} parquet files under {sample}")

    paths = []
    for i, name in enumerate(names):
        t0 = time.time()
        p = hf_hub_download(REPO_ID, name, repo_type="dataset",
                            local_dir=raw_dir)
        mb = os.path.getsize(p) / 2**20
        print(f"  [{i + 1}/{len(names)}] {os.path.basename(name)} "
              f"{mb:,.0f} MiB ({time.time() - t0:.0f}s)", flush=True)
        paths.append(p)
    return paths


def local_parquet(raw_dir: str) -> list:
    out = []
    for root, _, files in os.walk(raw_dir):
        out += [os.path.join(root, f) for f in files if f.endswith(".parquet")]
    return sorted(out)


# ---------------------------------------------------------------------
# stage 2: tokenize
# ---------------------------------------------------------------------
def tokenize(paths: list, out_dir: str, shard_tokens: int, val_tokens: int,
             threads: int, row_batch: int) -> dict:
    import pyarrow.parquet as pq
    import tiktoken

    os.makedirs(out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token                     # 50256, the document separator

    buf, buffered, shard_i = [], 0, 0
    written = {"train": 0, "val": 0}
    # The first val_tokens go to the validation shard; everything after
    # is training data. Held out means held out.
    target, kind = val_tokens, "val"
    t0 = time.time()

    def flush(kind: str) -> bool:
        """Write one shard. Returns False if it already existed."""
        nonlocal buf, buffered, shard_i
        if not buffered:
            return True
        name = "val.bin" if kind == "val" else f"train_{shard_i:04d}.bin"
        path = os.path.join(out_dir, name)
        fresh = not os.path.exists(path)
        if fresh:
            np.concatenate(buf).astype(np.uint16).tofile(path + ".part")
            os.replace(path + ".part", path)
        written[kind] += buffered
        rate = written["train"] + written["val"]
        print(f"  {name}: {human(buffered)} tokens"
              + ("" if fresh else "  (already present, skipped)")
              + f"   [{human(rate)} total, {time.time() - t0:.0f}s, "
                f"{rate / max(time.time() - t0, 1e-9) / 1e6:.2f}M tok/s]",
              flush=True)
        if kind == "train":
            shard_i += 1
        buf, buffered = [], 0
        return fresh

    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=row_batch, columns=["text"]):
            texts = batch.column("text").to_pylist()
            # tiktoken releases the GIL and threads this in Rust, which is
            # why this script has no multiprocessing in it.
            for ids in enc.encode_ordinary_batch(texts, num_threads=threads):
                buf.append(np.array([eot] + ids, dtype=np.uint16))
                buffered += len(ids) + 1
            while buffered >= target:
                flush(kind)
                target, kind = shard_tokens, "train"
    flush(kind)

    return {
        "tokenizer": "gpt2 (tiktoken)",
        "vocab_size": 50257,
        "dtype": "uint16",
        "eot_token": eot,
        "dataset": REPO_ID,
        "sample": SAMPLE,
        "train_tokens": written["train"],
        "val_tokens": written["val"],
        "train_shards": shard_i,
        "shard_tokens": shard_tokens,
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "build_seconds": round(time.time() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["download", "tokenize", "all"],
                    default="all")
    ap.add_argument("--raw", required=True, help="where the parquet lives")
    ap.add_argument("--out", help="where the .bin shards go (tokenize/all)")
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--max-files", type=int, default=0,
                    help="only the first N parquet files; 0 means all")
    ap.add_argument("--shard-tokens", type=int, default=250_000_000)
    ap.add_argument("--val-tokens", type=int, default=10_000_000)
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--row-batch", type=int, default=2048)
    args = ap.parse_args(argv)

    if args.stage in ("tokenize", "all") and not args.out:
        print("--out is required for tokenize", file=sys.stderr)
        return 2

    try:
        import pyarrow  # noqa: F401
        import tiktoken  # noqa: F401
    except ImportError:
        print("needs `tiktoken` and `pyarrow`. On Delta:\n"
              "  module load pytorch-conda/2.12 && conda activate base",
              file=sys.stderr)
        return 1

    if args.stage in ("download", "all"):
        download(args.raw, args.sample, args.max_files)
    if args.stage == "download":
        return 0

    paths = local_parquet(args.raw)
    if not paths:
        print(f"no parquet under {args.raw}; run --stage download first",
              file=sys.stderr)
        return 1
    print(f"tokenizing {len(paths)} files with {args.threads} threads")

    meta = tokenize(paths, args.out, args.shard_tokens, args.val_tokens,
                    args.threads, args.row_batch)
    meta["source_files"] = len(paths)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{human(meta['train_tokens'])} train "
          f"({meta['train_shards']} shards) + "
          f"{human(meta['val_tokens'])} val -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
