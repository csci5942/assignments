#!/usr/bin/env python3
"""Environment check for CSCI 5942 Assignment 4.

    python check_env.py

Prints a pass/fail line per requirement and, for anything that fails,
what to do about it. Exits nonzero if a required check fails.

Run it on a login node to check your setup, and again inside a job to
check the GPUs. The GPU and NCCL checks are skipped, not failed, when
you are on a login node -- there is no GPU there and that is fine.
"""

import importlib
import os
import shutil
import subprocess
import sys

OK, BAD, WARN, SKIP = "  ok  ", " FAIL ", " warn ", " skip "
_failures: list[str] = []
_warnings: list[str] = []

MODULE_FIX = ("module load pytorch-conda/2.12 && conda activate base\n"
              "       (do not build your own venv -- you lose the "
              "aws-ofi-nccl plugin Part 4 needs)")


def report(status: str, name: str, detail: str = "", fix: str = "") -> None:
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if status == BAD:
        _failures.append(f"{name}: {fix or detail}")
    elif status == WARN:
        _warnings.append(f"{name}: {fix or detail}")


def check_python() -> None:
    v = sys.version_info
    got = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 10):
        report(OK, "Python 3.10+", got)
    else:
        report(BAD, "Python 3.10+", f"found {got}", MODULE_FIX)


def check_module() -> None:
    """Are we inside the site PyTorch module rather than something else?"""
    loaded = os.environ.get("LOADEDMODULES", "")
    if "pytorch-conda" in loaded:
        version = os.environ.get("CONDA_PREFIX", "").split("/")[-1]
        report(OK, "pytorch-conda module", version or "loaded")
    else:
        report(WARN, "pytorch-conda module", "not loaded", MODULE_FIX)


def check_packages() -> None:
    for name, required in [("torch", True), ("numpy", True),
                           ("tiktoken", False), ("pytest", False)]:
        try:
            m = importlib.import_module(name)
            report(OK, name, getattr(m, "__version__", "ok"))
        except ImportError:
            report(BAD if required else WARN, name, "not importable", MODULE_FIX)


def check_gpu() -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        if shutil.which("srun") and not os.environ.get("SLURM_JOB_ID"):
            report(SKIP, "CUDA device", "login node; check again inside a job")
        else:
            report(BAD, "CUDA device", "torch.cuda.is_available() is False",
                   "request GPUs: --gpus-per-node=N")
        return
    n = torch.cuda.device_count()
    names = {torch.cuda.get_device_name(i) for i in range(n)}
    gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    report(OK, "CUDA device", f"{n} x {', '.join(names)}, {gib:.0f} GiB each")
    if not torch.cuda.is_bf16_supported():
        report(BAD, "bfloat16", "unsupported on this device",
               "use an A40, A100 or H200 partition")
    else:
        report(OK, "bfloat16", "supported")


def check_nccl() -> None:
    """The plugin that makes Part 4 fast rather than merely correct."""
    if os.environ.get("NCCL_NET_PLUGIN") == "ofi":
        report(OK, "aws-ofi-nccl plugin", "NCCL_NET_PLUGIN=ofi")
    else:
        report(WARN, "aws-ofi-nccl plugin", "not configured",
               "loaded automatically by pytorch-conda; without it, "
               "multi-node NCCL in Part 4 falls back to TCP sockets "
               "(slow, but not wrong)")


def check_corpus() -> None:
    from trainer.data import DEFAULT_ROOT
    root = os.environ.get("A4_DATA", DEFAULT_ROOT)
    if not os.path.isdir(root):
        report(BAD, "corpus", f"{root} not found",
               "the course corpus is on /work/nvme; if you are off-cluster, "
               "build one with tools/prepare_fineweb.py --docs 2000")
        return
    import glob
    shards = glob.glob(os.path.join(root, "train_*.bin"))
    val = os.path.exists(os.path.join(root, "val.bin"))
    tokens = sum(os.path.getsize(p) for p in shards) // 2
    if shards and val:
        report(OK, "corpus", f"{len(shards)} train shards, "
                             f"{tokens / 1e9:.1f}B tokens, val.bin present")
    else:
        report(BAD, "corpus", f"{len(shards)} train shards, val={val}",
               "expected train_*.bin and val.bin under " + root)


def check_slurm() -> None:
    if not shutil.which("sbatch"):
        report(WARN, "Slurm", "no sbatch on PATH", "are you on Delta?")
        return
    acct = subprocess.run(["sacctmgr", "-nP", "show", "assoc",
                           f"user={os.environ.get('USER','')}",
                           "format=Account"],
                          capture_output=True, text=True, timeout=30)
    accounts = {a.strip() for a in acct.stdout.split() if a.strip()}
    course = [a for a in accounts if a.startswith("bidb")]
    if course:
        report(OK, "Slurm account", ", ".join(sorted(course)))
    else:
        report(BAD, "Slurm account", f"no bidb-* account in {sorted(accounts)}",
               "you are not on the course allocation yet -- email the TA")


def main() -> int:
    print("CSCI 5942 Assignment 4 -- environment check\n")
    for fn in (check_python, check_module, check_packages, check_gpu,
               check_nccl, check_corpus, check_slurm):
        try:
            fn()
        except Exception as e:                       # noqa: BLE001
            report(BAD, fn.__name__, f"{type(e).__name__}: {e}")
    if _warnings:
        print("\nWarnings:")
        for w in _warnings:
            print(f"  - {w}")
    if _failures:
        print("\nFailures, and what to do about them:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("\nAll required checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
