"""DataLoader for A4. Fineweb dataset is already provided.
"""

import glob
import json
import os

import numpy as np
import torch

DEFAULT_ROOT = "/work/nvme/bidb/csci5942/a4/fineweb"


class TokenShards:
    """A set of .bin shards seen as one pool of sequence start positions.
    """

    def __init__(self, pattern: str, block_size: int):
        self.paths = sorted(glob.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(
                f"no shards matched {pattern!r}.\n"
                f"The course corpus lives at {DEFAULT_ROOT}. If you are "
                f"running somewhere else, pass --data-root, or build your "
                f"own with tools/prepare_fineweb.py."
            )
        self.block_size = block_size
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r")
                       for p in self.paths]
        # Valid start positions per shard, and their running total.
        starts = [max(0, len(s) - block_size - 1) for s in self.shards]
        self.starts = np.asarray(starts, dtype=np.int64)
        self.cumulative = np.cumsum(self.starts)
        self.n_positions = int(self.cumulative[-1])
        self.n_tokens = int(sum(len(s) for s in self.shards))
        if self.n_positions <= 0:
            raise ValueError(f"shards in {pattern!r} are shorter than "
                             f"block_size+1 = {block_size + 1} tokens")

    def __repr__(self):
        n = self.n_tokens
        size = f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"
        return f"TokenShards({len(self.paths)} shards, {size} tokens)"

    def take(self, positions: np.ndarray) -> torch.Tensor:
        """Gather sequences of block_size+1 tokens at the given positions.

        Returns an int64 tensor of shape (len(positions), block_size+1).
        """
        which = np.searchsorted(self.cumulative, positions, side="right")
        base = np.where(which > 0, self.cumulative[np.maximum(which - 1, 0)], 0)
        offsets = positions - base
        n = self.block_size + 1
        out = np.empty((len(positions), n), dtype=np.int64)
        for i, (shard_i, off) in enumerate(zip(which, offsets)):
            out[i] = self.shards[shard_i][off:off + n].astype(np.int64)
        return torch.from_numpy(out)


class Batcher:
    """Global batches, sliced by data parallel rank.

    A global batch is `global_batch` sequences of `block_size` tokens.
    The sequences for step `s` come from a generator seeded by
    `(seed, s)`, so any rank in any configuration can reconstruct them
    without talking to anyone.
    """

    def __init__(self, shards: TokenShards, global_batch: int, seed: int = 1337):
        self.shards = shards
        self.global_batch = global_batch
        self.seed = seed
        self.block_size = shards.block_size

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch * self.block_size

    def global_positions(self, step: int) -> np.ndarray:
        """The start positions of every sequence in global step `step`."""
        rng = np.random.default_rng([self.seed, step])
        return rng.integers(0, self.shards.n_positions,
                            size=self.global_batch, dtype=np.int64)

    def batch(self, step: int, dp_rank: int = 0, dp_world: int = 1):
        """This rank's slice of global step `step`.

        Returns (x, y), each (global_batch // dp_world, block_size).
        Split it into micro-batches yourself; that is a throughput
        decision, not a correctness one.
        """
        if self.global_batch % dp_world:
            raise ValueError(
                f"global_batch {self.global_batch} is not divisible by "
                f"dp_world {dp_world}; every rank must get the same count "
                f"or the gradient average is silently wrong.")
        per = self.global_batch // dp_world
        pos = self.global_positions(step)[dp_rank * per:(dp_rank + 1) * per]
        seq = self.shards.take(pos)
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


def micro_batch_seed(seed: int, step: int, micro_index: int) -> int:
    """A seed keyed to the *global* index of a micro-batch within a step.

    `Batcher` guarantees that global step `s` is the same sequences
    however you split them. That makes the *data* identical across
    parallelizations; it does not make the *computation* identical.
    Anything stochastic inside the model -- dropout, here -- draws from
    the global RNG, and a rank holding a quarter of the batch draws a
    quarter-sized mask from a different point in that stream than a
    single rank holding all of it. Same data, different masks, different
    gradients, and a correctness test that can never pass.

    Seeding per micro-batch fixes it. Number the micro-batches of a
    global step across the whole data parallel group: with `dp_world`
    ranks each running `accum` micro-batches, rank `r`'s `j`-th
    micro-batch has global index `r * accum + j`. Seed with this
    function before each forward pass and every configuration draws the
    same mask for the same sequences.

    The index has to be the *global* one. Passing `j` instead of
    `r * accum + j` makes every rank draw rank 0's masks, which looks
    fine -- the loss curve is smooth and the run converges -- and is a
    different computation from the one you are claiming to reproduce.
    """
    h = (int(seed) * 1_000_003 + int(step)) * 1_000_003 + int(micro_index)
    return h % (2 ** 31 - 1)


class ValBatcher:
    """Held-out loss, on fixed non-overlapping blocks.
    """

    def __init__(self, shards: TokenShards, batch_size: int, n_batches: int = 16):
        self.shards = shards
        self.batch_size = batch_size
        self.n_batches = n_batches
        stride = shards.block_size + 1
        total = batch_size * n_batches
        if total * stride > shards.n_positions:
            raise ValueError("validation shard is too small for "
                             f"{n_batches} batches of {batch_size}")
        self.positions = (np.arange(total, dtype=np.int64) * stride)

    def __iter__(self):
        for i in range(self.n_batches):
            pos = self.positions[i * self.batch_size:(i + 1) * self.batch_size]
            seq = self.shards.take(pos)
            yield seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


class Prefetcher:
    """Fetch step s+1 on a background thread while step s computes.

    Reading a batch means 128 random seeks into a memory-mapped file on a
    shared parallel filesystem. Measured on Delta that costs about 160 ms
    cold and 0.5 ms once the pages are resident -- which, against a step
    time of a couple of seconds, is a few percent of every measurement in
    this assignment, spent on something none of it is about.

    So the fetch happens on a thread. numpy drops the GIL for the page
    faults, so the main thread keeps the GPU busy while the next batch is
    pulled in. This is given, and it is also a small example of the thing
    Part 1 asks you to do to gradients: find the serial dependency, and
    overlap it with work that does not need it yet.
    """

    def __init__(self, batcher: "Batcher", start: int = 0, depth: int = 2,
                 dp_rank: int = 0, dp_world: int = 1):
        import queue
        import threading
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()

        def produce():
            step = start
            while not self.stop.is_set():
                item = batcher.batch(step, dp_rank, dp_world)
                while not self.stop.is_set():
                    try:
                        self.q.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                step += 1

        self.thread = threading.Thread(target=produce, daemon=True)
        self.thread.start()

    def next(self):
        return self.q.get()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_corpus(root: str, block_size: int, split: str = "train") -> TokenShards:
    pattern = os.path.join(root, "train_*.bin" if split == "train" else "val.bin")
    return TokenShards(pattern, block_size)


def corpus_meta(root: str) -> dict:
    path = os.path.join(root, "meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="peek at the corpus")
    ap.add_argument("--data-root", default=DEFAULT_ROOT)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--global-batch", type=int, default=8)
    args = ap.parse_args()

    shards = open_corpus(args.data_root, args.block_size)
    print(shards, corpus_meta(args.data_root))
    b = Batcher(shards, args.global_batch)
    x, y = b.batch(0)
    print("x", tuple(x.shape), "y", tuple(y.shape))
    print("targets are inputs shifted by one:",
          bool((x[0, 1:] == y[0, :-1]).all()))
    # The property the whole assignment rests on.
    whole = b.batch(7, dp_rank=0, dp_world=1)[0]
    halves = [b.batch(7, dp_rank=r, dp_world=2)[0] for r in range(2)]
    print("step 7 is the same batch at dp=1 and dp=2:",
          bool(torch.equal(whole, torch.cat(halves))))
