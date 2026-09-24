"""How fast can these GPUs add each other's numbers up?

Given. Every part of this assignment after this one is an argument about
whether some quantity of communication was worth it, and you cannot make
that argument without knowing what communication costs.

    srun python -m bench.allreduce
    srun python -m bench.allreduce --min-mb 1 --max-mb 1024 --csv bw.csv

**Bus bandwidth, not algorithm bandwidth.** The obvious number to report
is `N / t`: bytes in the buffer over seconds elapsed. That number is not
comparable across different rank counts, because a ring all-reduce over
`P` ranks does not move `N` bytes -- it moves each of `P` chunks around
the ring twice, once to reduce and once to gather, and every rank sends

    2 * (P - 1) / P * N

bytes to get there. Dividing *that* by the time gives bus bandwidth: the
traffic actually crossing the wire, which is the quantity a link has a
fixed budget of. It is what `nccl-tests` reports and what you should
compare against a link's specification.

Note what the factor does at the two ends. At `P = 2` it is 1, so bus
and algorithm bandwidth agree. As `P` grows it approaches 2, so a
four-rank all-reduce moves 1.5x the bytes a two-rank one does for the
same buffer -- which is most of why four ranks are not twice as fast as
two, and is worth having in hand before you are surprised by it in
Part 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.mesh import env_local_rank, init_distributed  # noqa: E402

MB = 1024 * 1024


def sizes_between(min_mb: int, max_mb: int) -> list[int]:
    """Powers of two from min_mb to max_mb, in bytes."""
    out, n = [], min_mb
    while n <= max_mb:
        out.append(n * MB)
        n *= 2
    return out


def bus_bandwidth(nbytes: int, seconds: float, world: int) -> float:
    """Bytes per second actually crossing the wire, per rank."""
    return 2.0 * (world - 1) / world * nbytes / seconds


def time_all_reduce(buf: torch.Tensor, iters: int, warmup: int,
                    device: str) -> float:
    """Median seconds for one all-reduce of `buf`.

    Median rather than mean: on a shared machine one iteration in twenty
    lands behind someone else's traffic, and a mean lets that one sample
    move the answer. Every rank must finish before the clock stops, so
    there is a barrier before the timer starts and a synchronize after
    each call -- without them you are timing the enqueue, not the
    collective, and the numbers come out impossibly good.
    """
    cuda = device.startswith("cuda")
    for _ in range(warmup):
        dist.all_reduce(buf)
    if cuda:
        torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(buf)
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-mb", type=int, default=1)
    ap.add_argument("--max-mb", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--csv", default=None,
                    help="append results here, for the plot")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if not init_distributed():
        print("bench.allreduce needs more than one rank.\n"
              "  srun --ntasks=4 --gpus-per-node=4 python -m bench.allreduce\n"
              "or use one of the scripts in slurm/.", file=sys.stderr)
        return 2

    rank, world = dist.get_rank(), dist.get_world_size()
    cuda = torch.cuda.is_available()
    device = f"cuda:{env_local_rank()}" if cuda else "cpu"
    if cuda:
        torch.cuda.set_device(env_local_rank())
    dtype = getattr(torch, args.dtype)
    lead = rank == 0

    if lead:
        name = torch.cuda.get_device_name(device) if cuda else "cpu"
        print(f"all-reduce over {world} ranks on {name}, {args.dtype}, "
              f"{args.iters} iterations (median)")
        print(f"  bus factor 2(P-1)/P = {2 * (world - 1) / world:.3f}")
        print()
        print(f"  {'size':>10}  {'time':>10}  {'algbw':>12}  {'busbw':>12}")
        print(f"  {'':>10}  {'':>10}  {'GB/s':>12}  {'GB/s':>12}")

    rows = []
    for nbytes in sizes_between(args.min_mb, args.max_mb):
        n = nbytes // torch.tensor([], dtype=dtype).element_size()
        try:
            buf = torch.ones(n, dtype=dtype, device=device)
        except torch.cuda.OutOfMemoryError:
            if lead:
                print(f"  {nbytes // MB:>8} MB  out of memory, stopping")
            break

        seconds = time_all_reduce(buf, args.iters, args.warmup, device)
        algbw = nbytes / seconds
        busbw = bus_bandwidth(nbytes, seconds, world)
        rows.append({"world_size": world, "bytes": nbytes,
                     "mb": nbytes // MB, "dtype": args.dtype,
                     "seconds": seconds, "algbw_gbps": algbw / 1e9,
                     "busbw_gbps": busbw / 1e9})
        if lead:
            print(f"  {nbytes // MB:>7} MB  {seconds * 1e3:>8.3f} ms  "
                  f"{algbw / 1e9:>12.2f}  {busbw / 1e9:>12.2f}")

        del buf
        if cuda:
            torch.cuda.empty_cache()

    if lead and rows:
        peak = max(rows, key=lambda r: r["busbw_gbps"])
        print(f"\n  peak bus bandwidth {peak['busbw_gbps']:.2f} GB/s "
              f"at {peak['mb']} MB")
        print("  Small buffers are latency-bound and large ones are "
              "bandwidth-bound;\n  the number to quote in later parts is "
              "the plateau, not the peak of a\n  single point.")

        if args.csv:
            import csv
            exists = os.path.exists(args.csv)
            with open(args.csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                if not exists:
                    w.writeheader()
                w.writerows(rows)
            print(f"  -> {args.csv}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"  -> {args.json}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
