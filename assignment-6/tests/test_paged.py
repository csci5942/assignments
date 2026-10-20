"""Self-checks for the paged KV cache (Part 2).

    python -m pytest tests/ -q

CPU, fp32, a small random model. The block size is small (4) so that
every sequence spans several blocks and sequences decoded together
interleave in the pool.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig  # noqa: E402
from paged import BlockAllocator, PagedKVCache, generate  # noqa: E402

CFG = GPTConfig(block_size=64, vocab_size=97, n_layer=2, n_head=4, n_embd=64,
                dropout=0.0, bias=True)
ATOL = 1e-4
BS = 4


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(CFG).eval()


def prompts(*lens, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, CFG.vocab_size, (n,), generator=g).tolist() for n in lens]


def test_allocator():
    a = BlockAllocator(5)
    assert a.num_free == 5
    got = [a.allocate() for _ in range(5)]
    assert sorted(got) == [0, 1, 2, 3, 4], "allocate must hand out each block once"
    assert a.num_free == 0
    with pytest.raises(MemoryError):
        a.allocate()
    a.free(got[1:3])
    assert a.num_free == 2
    assert sorted([a.allocate(), a.allocate()]) == sorted(got[1:3])


def test_begin_allocates_and_maps_slots():
    cache = PagedKVCache(CFG, num_blocks=16, block_size=BS)
    cache.allocator.allocate()               # block tables need not start at 0
    cache.add(0)
    cache.add(1)
    cache.select([0, 1])
    pos = cache.begin(6)
    cache.end(6)
    assert pos.tolist() == [list(range(6))] * 2
    for s in (0, 1):
        assert len(cache.tables[s]) == 2, "6 tokens in blocks of 4 need 2 blocks"
    assert set(cache.tables[0]).isdisjoint(cache.tables[1])
    for b, s in enumerate((0, 1)):
        t = cache.tables[s]
        want = [t[j // BS] * BS + j % BS for j in range(6)]
        assert cache.slots[b].tolist() == want, "slot = table[pos // bs] * bs + pos % bs"
    pos = cache.begin(3)                      # 6 + 3 = 9 tokens: a third block
    assert pos.tolist() == [[6, 7, 8]] * 2
    assert all(len(cache.tables[s]) == 3 for s in (0, 1))
    assert cache.allocator.num_free == 16 - 1 - 6


def test_prefill_matches_full_forward(model):
    [p] = prompts(11)
    x = torch.tensor([p])
    with torch.no_grad():
        ref, _ = model(x)
        cache = PagedKVCache(CFG, 16, BS)
        cache.add(0)
        cache.select([0])
        out, _ = model(x, cache=cache)
    assert cache.lens[0] == 11
    assert torch.allclose(out, ref, atol=ATOL), "paged prefill logits differ from the uncached forward"


def test_ragged_decode_matches_each_alone(model):
    """Three sequences of different lengths, prefilled one at a time,
    then decoded together, one token per step, with no padding."""
    ps = prompts(3, 9, 5)
    full = [torch.tensor([p + prompts(8, seed=10 + i)[0]]) for i, p in enumerate(ps)]
    with torch.no_grad():
        refs = [model(x)[0][0] for x in full]
        cache = PagedKVCache(CFG, 32, BS)
        for s, p in enumerate(ps):
            cache.add(s)
            cache.select([s])
            model(torch.tensor([p]), cache=cache)
        cache.select([0, 1, 2])
        for t in range(8):
            ids = torch.tensor([[x[0, len(p) + t].item()] for x, p in zip(full, ps)])
            out, _ = model(ids, cache=cache)
            for b, p in enumerate(ps):
                assert torch.allclose(out[b, 0], refs[b][len(p) + t], atol=ATOL), \
                    f"sequence {b} (prompt {len(p)}) differs at decode step {t}: " \
                    "check the slots, the gather order, and the mask"


def test_generate_matches_uncached_and_frees(model):
    ps = prompts(2, 13, 7, 1)
    cache = PagedKVCache(CFG, 40, BS)
    out, peak = generate(model, ps, 12, cache)
    for b, p in enumerate(ps):
        ref = model.generate(torch.tensor([p]), 12, temperature=0)[0, len(p):]
        assert out[b] == ref.tolist(), f"sequence {b} differs from uncached generate()"
    assert peak["used"] == sum(len(p) + 11 for p in ps)
    assert peak["reserved"] == sum(-(-(len(p) + 11) // BS) * BS for p in ps)
    assert cache.allocator.num_free == 40, "remove() must return every block"
    assert not cache.tables and not cache.lens


def test_out_of_blocks(model):
    cache = PagedKVCache(CFG, 2, BS)
    cache.add(0)
    cache.select([0])
    with torch.no_grad():
        model(torch.zeros(1, 8, dtype=torch.long), cache=cache)
        with pytest.raises(MemoryError):
            model(torch.zeros(1, 1, dtype=torch.long), cache=cache)


def test_chunked_prefill(model):
    """Several new tokens at once on top of cached ones: the mask must be
    causal within the chunk and open to everything before it."""
    [p] = prompts(19)
    x = torch.tensor([p])
    with torch.no_grad():
        ref, _ = model(x)
        cache = PagedKVCache(CFG, 16, BS)
        cache.add(0)
        cache.select([0])
        a, _ = model(x[:, :6], cache=cache)
        b, _ = model(x[:, 6:], cache=cache)
    assert torch.allclose(torch.cat([a, b], 1), ref, atol=ATOL), \
        "second chunk differs: the mask for T > 1 new tokens on a non-empty sequence is off"


def test_past_max_len_raises():
    """A sequence may not grow past the model's context (block_size)."""
    cache = PagedKVCache(CFG, 64, BS)
    cache.add(0)
    cache.select([0])
    with pytest.raises(ValueError):
        cache.begin(CFG.block_size + 1)
