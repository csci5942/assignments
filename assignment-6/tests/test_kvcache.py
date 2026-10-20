"""Self-checks for the KV cache (Part 1).

    python -m pytest tests/ -q

They run on CPU in a few seconds, on a small random model in fp32. Every
check compares the cached path against A1's uncached forward, which you
trust: same logits to 1e-4, same greedy tokens.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig, KVCache, left_pad  # noqa: E402

CFG = GPTConfig(block_size=64, vocab_size=97, n_layer=2, n_head=4, n_embd=64,
                dropout=0.0, bias=True)
ATOL = 1e-4


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(CFG).eval()


def test_update_stores_and_returns_prefix():
    cache = KVCache(CFG, batch_size=2, max_len=10)
    hd = CFG.n_embd // CFG.n_head
    k1, v1 = torch.randn(2, CFG.n_head, 3, hd), torch.randn(2, CFG.n_head, 3, hd)
    assert cache.begin(3).tolist() == [0, 1, 2]
    K, V, mask = cache.update(0, k1, v1)
    assert K.shape == (2, CFG.n_head, 3, hd)
    assert torch.equal(K, k1) and torch.equal(V, v1)
    assert torch.equal(mask.reshape(-1, 3, 3)[0], torch.tril(torch.ones(3, 3)).bool()), \
        "prefill mask must be causal"
    cache.end(3)
    k2, v2 = torch.randn(2, CFG.n_head, 1, hd), torch.randn(2, CFG.n_head, 1, hd)
    assert cache.begin(1).tolist() == [3], "begin returns positions pos .. pos+T-1: check the position ids"
    K, V, mask = cache.update(0, k2, v2)
    assert K.shape == (2, CFG.n_head, 4, hd), "update must return slots 0 .. pos+T-1"
    assert torch.equal(K[:, :, :3], k1) and torch.equal(K[:, :, 3:], k2)
    assert torch.equal(V[:, :, 3:], v2)
    assert mask.reshape(-1, 1, 4)[0].all(), "a new token sees every filled slot"
    assert cache.pos == 3, "update must not advance pos; end() does"


def test_prefill_matches_full_forward(model):
    x = torch.randint(0, CFG.vocab_size, (2, 12))
    with torch.no_grad():
        ref, _ = model(x)
        cache = KVCache(CFG, 2, 20)
        out, _ = model(x, cache=cache)
    assert cache.pos == 12, "forward must advance cache.pos by T"
    assert torch.allclose(out, ref, atol=ATOL), "prefill logits differ from the uncached forward"


def test_decode_matches_full_forward(model):
    """Prefill 5 tokens, then feed the rest one at a time."""
    x = torch.randint(0, CFG.vocab_size, (2, 16))
    with torch.no_grad():
        ref, _ = model(x)
        cache = KVCache(CFG, 2, 16)
        model(x[:, :5], cache=cache)
        for t in range(5, 16):
            out, _ = model(x[:, t:t + 1], cache=cache)
            assert torch.allclose(out[:, 0], ref[:, t], atol=ATOL), \
                f"decode logits at position {t} differ: check the position ids and the cached keys"


def test_chunked_prefill(model):
    """Several new tokens at once on top of a filled cache: the mask must be
    causal within the chunk and open to everything before it."""
    x = torch.randint(0, CFG.vocab_size, (1, 20))
    with torch.no_grad():
        ref, _ = model(x)
        cache = KVCache(CFG, 1, 20)
        a, _ = model(x[:, :8], cache=cache)
        b, _ = model(x[:, 8:20], cache=cache)
    assert torch.allclose(torch.cat([a, b], 1), ref, atol=ATOL), \
        "second chunk differs: the mask for T > 1 new tokens at pos > 0 is off"


def test_cache_full_raises(model):
    cache = KVCache(CFG, 1, 4)
    with torch.no_grad():
        model(torch.zeros(1, 4, dtype=torch.long), cache=cache)
        with pytest.raises(ValueError):
            model(torch.zeros(1, 1, dtype=torch.long), cache=cache)


def test_greedy_generate_matches_uncached(model):
    x = torch.randint(0, CFG.vocab_size, (3, 7))
    a = model.generate(x, 20, temperature=0)
    b = model.generate(x, 20, temperature=0, use_cache=True)
    assert torch.equal(a, b), "cached greedy tokens differ from uncached generate()"


def test_left_padded_logits(model):
    """Logits, not just tokens: every real position of a padded row equals
    the unpadded forward, and nothing is NaN."""
    torch.manual_seed(2)
    prompts = [torch.randint(1, CFG.vocab_size, (n,)).tolist() for n in (4, 9)]
    idx, pad = left_pad(prompts)
    with torch.no_grad():
        cache = KVCache(CFG, 2, idx.size(1), pad=pad)
        out, _ = model(idx, cache=cache)
        assert not torch.isnan(out).any(), \
            "NaN logits"
        for b, p in enumerate(prompts):
            ref, _ = model(torch.tensor([p]))
            assert torch.allclose(out[b, pad[b]:], ref[0], atol=ATOL)


def test_left_padded_batch_matches_each_alone(model):
    torch.manual_seed(1)
    prompts = [torch.randint(1, CFG.vocab_size, (n,)).tolist() for n in (3, 11, 6, 1)]
    idx, pad = left_pad(prompts)
    out = model.generate(idx, 15, temperature=0, use_cache=True, pad=pad)
    assert not torch.isnan(out.float()).any()
    for b, p in enumerate(prompts):
        alone = model.generate(torch.tensor([p]), 15, temperature=0)
        assert torch.equal(out[b, idx.size(1):], alone[0, len(p):]), \
            f"row {b} (prompt length {len(p)}) differs from running it alone: " \
            "check its position ids and the pad mask"
