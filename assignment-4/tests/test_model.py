"""The given model. If these fail, something is wrong with the handout,
not with you -- open an issue.

They are here because Parts 2 and 3 rewrite pieces of this model, and
you will want a baseline that you know was correct before you started.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from trainer.model import GPT, Block, GPTConfig  # noqa: E402


def test_shapes(tiny_config):
    model = GPT(tiny_config)
    x = torch.randint(0, tiny_config.vocab_size, (3, tiny_config.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (3, tiny_config.block_size, tiny_config.vocab_size)
    assert loss.ndim == 0


def test_initial_loss_is_uniform(tiny_config):
    """A freshly initialized model should be maximally unsure.

    Cross-entropy against a uniform distribution over V tokens is ln(V).
    Being far off means the initialization is broken; this is the
    cheapest sanity check in all of language modelling, and it is worth
    running on your own models for the rest of your life.
    """
    torch.manual_seed(0)
    model = GPT(tiny_config)
    x = torch.randint(0, tiny_config.vocab_size, (8, tiny_config.block_size))
    _, loss = model(x, x)
    assert loss.item() == pytest.approx(math.log(tiny_config.vocab_size),
                                        abs=0.15)


def test_causality(tiny_config):
    """Position t must not see position t+1.

    Perturb the last token and check that no earlier logit moves. This
    is A1's test, and it still catches the same bugs after Part 2
    reshards attention across ranks.
    """
    torch.manual_seed(0)
    model = GPT(tiny_config).eval()
    x = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.block_size))
    with torch.no_grad():
        base, _ = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % tiny_config.vocab_size
        other, _ = model(x2)
    assert torch.allclose(base[0, :-1], other[0, :-1], atol=1e-5), \
        "changing the last token changed an earlier position's logits"


def test_blocks_are_separable(tiny_config):
    """Part 3 cuts the model between blocks, so nothing may span one."""
    model = GPT(tiny_config)
    assert all(isinstance(b, Block) for b in model.transformer.h)
    x = torch.randn(2, tiny_config.block_size, tiny_config.n_embd)
    for b in model.transformer.h:
        x = b(x)
    assert x.shape == (2, tiny_config.block_size, tiny_config.n_embd)


def test_param_accounting(tiny_config):
    model = GPT(tiny_config)
    assert model.num_params() == sum(p.numel() for p in model.parameters())
    buckets = model.param_bytes()
    # params + grads + two AdamW moments, fp32
    assert sum(buckets.values()) == 16 * model.num_params()


def test_config_ladder_is_wellformed():
    """Every shipped config must build, and heads must divide the width.

    The second half matters in Part 2: tensor parallelism splits
    attention by head, so `n_head` also has to divide the tensor
    parallel degree you choose.
    """
    import glob
    import json
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configs = sorted(glob.glob(os.path.join(here, "configs", "*.json")))
    assert configs, "no configs found"
    for path in configs:
        with open(path) as f:
            raw = json.load(f)
        cfg = GPTConfig.from_dict(raw)
        assert cfg.n_embd % cfg.n_head == 0, path
        assert raw["global_batch"] % raw["micro_batch"] == 0, path
