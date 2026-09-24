"""The loader, and the one property the rest of the assignment rests on."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trainer.data import Batcher, ValBatcher, open_corpus  # noqa: E402

BLOCK = 64


def test_targets_are_inputs_shifted_by_one(corpus):
    b = Batcher(open_corpus(corpus, BLOCK), global_batch=8)
    x, y = b.batch(0)
    assert x.shape == (8, BLOCK) and y.shape == (8, BLOCK)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batches_are_deterministic(corpus):
    shards = open_corpus(corpus, BLOCK)
    a = Batcher(shards, global_batch=8, seed=1337).batch(3)[0]
    b = Batcher(shards, global_batch=8, seed=1337).batch(3)[0]
    assert torch.equal(a, b)


def test_different_steps_differ(corpus):
    b = Batcher(open_corpus(corpus, BLOCK), global_batch=8)
    assert not torch.equal(b.batch(0)[0], b.batch(1)[0])


@pytest.mark.parametrize("dp_world", [1, 2, 4, 8])
def test_global_batch_is_independent_of_dp_degree(corpus, dp_world):
    """The property Part 4's loss gate depends on.

    Step 5 is the same 8 sequences however many data parallel ranks
    split it. If this ever stops being true, a fast configuration and a
    correct one stop being the same thing, and nothing in Part 4 can be
    scored.
    """
    b = Batcher(open_corpus(corpus, BLOCK), global_batch=8)
    whole = b.batch(5)[0]
    pieces = [b.batch(5, dp_rank=r, dp_world=dp_world)[0]
              for r in range(dp_world)]
    assert torch.equal(whole, torch.cat(pieces))
    assert all(len(p) == 8 // dp_world for p in pieces)


def test_uneven_dp_split_is_refused(corpus):
    b = Batcher(open_corpus(corpus, BLOCK), global_batch=8)
    with pytest.raises(ValueError, match="divisible"):
        b.batch(0, dp_rank=0, dp_world=3)


def test_val_is_fixed_and_non_overlapping(corpus):
    shards = open_corpus(corpus, BLOCK, "val")
    first = [x for x, _ in ValBatcher(shards, 4, 3)]
    again = [x for x, _ in ValBatcher(shards, 4, 3)]
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    flat = torch.cat(first)
    starts = ValBatcher(shards, 4, 3).positions
    assert len(np.unique(starts)) == len(starts)
    assert len(flat) == 12
