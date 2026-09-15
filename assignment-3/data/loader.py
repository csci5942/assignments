#!/usr/bin/env python3
"""Stream a training mix off disk.

Part 2 of Assignment 3. **One function is yours: `MixDataset.__iter__`.**
Everything else in this file is given.

    python -m pytest tests/test_loader.py -q
    python data/loader.py --mix out/raw/mix_baseline --steps 20

A mix is a directory of Parquet shards holding text. Your job is to turn
that into fixed-length batches of token ids, reading each shard exactly
once across all workers and all ranks.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_vocab(path=None):
    """Assignment 1's 65-character vocabulary."""
    path = path or os.path.join(HERE, "vocab.json")
    with open(path) as f:
        itos = json.load(f)["itos"]
    return {c: i for i, c in enumerate(itos)}, itos


# =====================================================================
# yours
# =====================================================================
def shard_paths(mix_dir: str) -> list:
    """Every Parquet shard of a mix, in a stable order.

    In:  `mix_dir`, e.g. out/raw/mix_baseline
    Out: a sorted list of absolute paths.
    """
    return sorted(os.path.abspath(p)
                  for p in glob.glob(os.path.join(mix_dir, "*.parquet")))


def assign_shards(paths: list, rank: int, world: int,
                  worker: int, workers: int) -> list:
    """Which shards this (rank, worker) is responsible for.

    In:  all shard paths; the distributed rank out of `world`; the
         DataLoader worker id out of `workers`.
    Out: this reader's subset. Every shard must go to exactly ONE
         reader, and the union over all (rank, worker) pairs must be the
         whole list.
    """
    total = world * workers
    me = rank * workers + worker
    return [p for i, p in enumerate(paths) if i % total == me]


def encode(text: str, stoi: dict) -> list:
    """Text -> token ids, dropping anything outside the vocabulary.
    """
    return [stoi[c] for c in text if c in stoi]


def pack(token_stream, block_size: int):
    """Turn a stream of variable-length token lists into fixed blocks.

    In:  an iterable of token-id lists (one per line of the mix)
    Out: a generator of lists, each exactly `block_size + 1` long.

    The model needs fixed-length windows, but lines are all different
    lengths. So accumulate into a buffer and emit a block whenever you
    have enough, keeping the remainder for the next one.
    """
    buf = []
    for ids in token_stream:
        buf.extend(ids)
        while len(buf) >= block_size + 1:
            yield buf[:block_size + 1]
            buf = buf[block_size:]  


class MixDataset(torch.utils.data.IterableDataset):
    """Stream a mix's shards as fixed-length training blocks."""

    def __init__(self, mix_dir, block_size=256, rank=0, world=1,
                 stoi=None, shuffle_shards=True, seed=1337):
        self.mix_dir = mix_dir
        self.block_size = block_size
        self.rank, self.world = rank, world
        self.stoi = stoi or load_vocab()[0]
        self.shuffle_shards, self.seed = shuffle_shards, seed
        self.paths = shard_paths(mix_dir)
        if not self.paths:
            raise FileNotFoundError(f"no .parquet shards in {mix_dir}")

    def _my_paths(self):
        info = torch.utils.data.get_worker_info()
        worker = info.id if info else 0
        workers = info.num_workers if info else 1
        mine = assign_shards(self.paths, self.rank, self.world, worker, workers)
        if self.shuffle_shards:
            # Shard ORDER is shuffled, not shard membership: which reader
            # owns which shard must stay fixed or the split breaks.
            import random
            random.Random(self.seed + self.rank * 1000 + worker).shuffle(mine)
        return mine

    def __iter__(self):
        """Yield (input, target) block pairs for this worker's shards.

        The only function in this file you write. A generator: yield as
        you go, holding one shard in memory at a time.

        Take your shards from `self._my_paths()`,
        and for each one read its `text`
        column, `encode` each row, and `pack` the stream into blocks. A
        block is `block_size + 1` tokens; yield it as `(x, y)`, both
        `block_size` long and `torch.int64`, with `y` being `x` shifted
        one step left.
        """
        import pyarrow.parquet as pq
        # TODO: implement this stage.
        raise NotImplementedError("stage not implemented")


# =====================================================================
# given -- a self-test you can run directly
# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    stoi, itos = load_vocab()
    ds = MixDataset(args.mix, block_size=args.block_size, stoi=stoi)
    print(f"{len(ds.paths)} shards in {args.mix}")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                     num_workers=args.workers)
    seen = 0
    for i, (x, y) in enumerate(dl):
        if i == 0:
            print(f"x {tuple(x.shape)} y {tuple(y.shape)} dtype {x.dtype}")
            print("decoded:", repr("".join(itos[t] for t in x[0][:80].tolist())))
            assert torch.equal(x[0][1:], y[0][:-1]), \
                "y must be x shifted by one"
        seen += x.shape[0]
        if i + 1 >= args.steps:
            break
    print(f"{seen} sequences over {args.steps} batches, "
          f"{seen * args.block_size:,} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
