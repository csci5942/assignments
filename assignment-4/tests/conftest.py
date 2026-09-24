"""Shared fixtures.

Every test here runs on a CPU in a couple of seconds, with no corpus and
no GPU, so you can run the suite while you are writing code rather than
only when you have an allocation. The trick is a synthetic corpus: the
model does not care whether its tokens mean anything, and correctness
does not either.
"""

import multiprocessing
import os
import socket
import sys
import traceback

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

# A vocabulary small enough that a 50257-wide output head does not make
# the suite unbearable. Nothing under test depends on the real size.
TEST_VOCAB = 512


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """Two train shards and a val shard of random tokens."""
    root = tmp_path_factory.mktemp("corpus")
    rng = np.random.default_rng(0)
    for i in range(2):
        rng.integers(0, TEST_VOCAB, size=20_000,
                     dtype=np.uint16).tofile(root / f"train_{i:04d}.bin")
    rng.integers(0, TEST_VOCAB, size=10_000,
                 dtype=np.uint16).tofile(root / "val.bin")
    return str(root)


@pytest.fixture
def tiny_config():
    """A model small enough to train a few steps on a laptop."""
    from trainer.model import GPTConfig
    return GPTConfig(block_size=64, vocab_size=TEST_VOCAB, n_layer=2,
                     n_head=4, n_embd=64, dropout=0.0)


# ---------------------------------------------------------------------
# running several ranks on one machine, with no GPU
# ---------------------------------------------------------------------
# Parts 1 through 3 are about collectives, and a collective needs more
# than one process. `gloo` gives us that on a CPU: the tests below fork
# four ranks, run them against each other, and report whichever rank
# failed first with its own traceback. Nothing here needs a GPU, a
# corpus, or an allocation.


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _rank_main(rank, world, port, backend, fn, args, errors):
    """One rank: join the group, run the body, report what went wrong."""
    import torch
    import torch.distributed as dist

    # One thread per rank. Torch otherwise sizes its intra-op pool to the
    # whole machine, and four ranks each claiming every core spend more
    # time contending than computing -- on a 64-core node that turned a
    # 40-second suite into a 10-minute one. Gloo also busy-waits, so the
    # oversubscription compounds.
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK="0",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        dist.init_process_group(backend, rank=rank, world_size=world)
    except Exception:                                        # noqa: BLE001
        errors.put((rank, traceback.format_exc()))
        return
    try:
        fn(rank, world, *args)
    except Exception:                                        # noqa: BLE001
        errors.put((rank, traceback.format_exc()))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:                                    # noqa: BLE001
            pass


def run_distributed(fn, world_size=4, *args, timeout=300, backend="gloo"):
    """Run `fn(rank, world, *args)` in `world_size` processes over gloo.

    Assertions belong *inside* `fn`, on every rank. The first rank to
    raise brings its traceback back here and fails the test with it, so
    a failure message written for a student still reads like one.

    Forked rather than spawned: a fresh interpreter would re-import torch
    in every child and turn a two-second test into a twelve-second one.
    Forking is safe here only because none of this touches CUDA.
    """
    __tracebackhide__ = True        # show the rank's traceback, not this one
    ctx = multiprocessing.get_context(
        "fork" if sys.platform.startswith("linux") else "spawn")
    errors = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_rank_main,
                         args=(r, world_size, port, backend, fn, args, errors),
                         daemon=True)
             for r in range(world_size)]
    for p in procs:
        p.start()

    failures, timed_out = [], []
    for r, p in enumerate(procs):
        p.join(timeout)
        if p.is_alive():
            timed_out.append(r)
            p.terminate()
            p.join(5)
    while not errors.empty():
        failures.append(errors.get())

    if timed_out:
        raise AssertionError(
            f"ranks {timed_out} were still running after {timeout}s. A "
            f"collective that some ranks enter and others do not will hang "
            f"like this rather than fail -- check that every rank calls "
            f"every all-reduce, in the same order, the same number of times."
            + (f"\n\nRank {failures[0][0]} also failed:\n{failures[0][1]}"
               if failures else ""))
    if failures:
        rank, tb = sorted(failures)[0]
        others = sorted({r for r, _ in failures} - {rank})
        also = f"  (ranks {others} failed too)" if others else ""
        raise AssertionError(f"rank {rank} failed:{also}\n\n{tb}")

    bad = [(r, p.exitcode) for r, p in enumerate(procs) if p.exitcode]
    if bad:
        raise AssertionError(f"ranks exited nonzero: {bad}")


@pytest.fixture
def distributed():
    """`distributed(fn, world_size, *args)` -- see `run_distributed`."""
    return run_distributed
