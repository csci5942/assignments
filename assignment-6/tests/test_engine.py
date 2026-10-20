"""Self-checks for the scheduler (Part 3).

    python -m pytest tests/ -q

CPU, fp32, a small random model.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Engine, Request, Scheduler, StaticScheduler  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from paged import PagedKVCache  # noqa: E402

CFG = GPTConfig(block_size=64, vocab_size=97, n_layer=2, n_head=4, n_embd=64,
                dropout=0.0, bias=True)
BS = 4


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(CFG).eval()


def requests(spec, seed=1):
    """spec: [(prompt_len, max_new), ...]"""
    g = torch.Generator().manual_seed(seed)
    return [Request(i, torch.randint(0, CFG.vocab_size, (p,), generator=g).tolist(), n)
            for i, (p, n) in enumerate(spec)]


def solo(model, r):
    out = model.generate(torch.tensor([r.prompt]), r.max_new, temperature=0)
    return out[0, len(r.prompt):].tolist()


def test_retire_frees_blocks():
    cache = PagedKVCache(CFG, 10, BS)
    s = Scheduler(cache, max_batch=4)
    [r] = requests([(5, 3)])
    s.add(r)
    assert s.schedule(0) == [r] and r.admitted == 0
    cache.add(r.id)
    cache.tables[r.id] = [cache.allocator.allocate() for _ in range(2)]
    r.out = [1, 2, 3]
    assert s.schedule(1) == []
    assert s.running == [] and s.finished == [r] and r.finished == 1
    assert cache.allocator.num_free == 10, "a finished request's blocks go back to the pool"


def test_admission_fcfs_and_safe():
    """Pool of 11 blocks of 4. Full lengths: 16 tokens = 4 blocks, 24 = 6,
    8 = 2, 4 = 1. The first two fit (10 blocks). The third does not fit in
    the 1 block left, and the fourth, which would, must wait behind it:
    FCFS does not skip ahead."""
    cache = PagedKVCache(CFG, 11, BS)
    s = Scheduler(cache, max_batch=8)
    rs = requests([(8, 8), (16, 8), (4, 4), (2, 2)])
    for r in rs:
        s.add(r)
    assert s.schedule(0) == rs[:2], "admit in order while the pool can hold each at full length"
    assert list(s.waiting) == rs[2:]


def test_admission_counts_blocks_already_held():
    """A running request that already holds some of its blocks is owed
    only the rest. Pool 10: r0's full length (16 tokens) is 4 blocks, all
    held already; r1's (24) is 6. r1 fits."""
    cache = PagedKVCache(CFG, 10, BS)
    s = Scheduler(cache, max_batch=8)
    r0, r1 = requests([(8, 8), (16, 8)])
    s.add(r0)
    s.schedule(0)
    cache.add(r0.id)
    cache.tables[r0.id] = [cache.allocator.allocate() for _ in range(4)]
    s.add(r1)
    assert s.schedule(1) == [r1], \
        "count only the blocks a running request still needs, not the ones it holds"


def test_max_batch():
    cache = PagedKVCache(CFG, 100, BS)
    s = Scheduler(cache, max_batch=2)
    rs = requests([(4, 4)] * 3)
    for r in rs:
        s.add(r)
    assert s.schedule(0) == rs[:2]


def test_continuous_refills_a_slot_immediately(model):
    """max_batch 2, output lengths 2, 12, 3: request 2 takes request 0's
    slot as soon as request 0 finishes, long before request 1 does."""
    rs = requests([(5, 2), (5, 12), (5, 3)])
    cache = PagedKVCache(CFG, 64, BS)
    Engine(model, cache, Scheduler(cache, max_batch=2)).run(rs)
    assert rs[2].admitted == rs[0].finished, "admit in the same iteration a slot frees"
    assert rs[2].admitted < rs[1].finished
    static = requests([(5, 2), (5, 12), (5, 3)])
    cache = PagedKVCache(CFG, 64, BS)
    Engine(model, cache, StaticScheduler(cache, max_batch=2)).run(static)
    assert static[2].admitted == static[1].finished    # static waits for the whole batch


def test_engine_outputs_match_solo(model):
    """A tight pool forces requests to wait; every request still gets
    exactly its solo greedy tokens, and every block comes back."""
    rs = requests([(9, 6), (3, 14), (12, 4), (6, 9), (2, 3), (10, 10)])
    cache = PagedKVCache(CFG, 12, BS)
    done = Engine(model, cache, Scheduler(cache, max_batch=4)).run(rs)
    assert sorted(r.id for r in done) == list(range(len(rs)))
    for r in rs:
        assert r.out == solo(model, r), f"request {r.id} differs from generating it alone"
    assert cache.allocator.num_free == 12


def test_optimistic_admission_with_preemption(model):
    """Optimistic can_admit on a pool too small for every request at once:
    the engine has to preempt (given), and every request must still come
    out exactly as if generated alone."""
    rs = requests([(9, 20), (3, 24), (12, 16), (6, 20)])
    cache = PagedKVCache(CFG, 14, BS)
    sched = Scheduler(cache, max_batch=4, optimistic=True)
    Engine(model, cache, sched).run(rs)
    assert sum(r.preemptions for r in rs) > 0, "this pool is too small not to preempt"
    for r in rs:
        assert r.out == solo(model, r), f"request {r.id} differs after preemption and recompute"
    assert cache.allocator.num_free == 14
