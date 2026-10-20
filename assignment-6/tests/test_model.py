"""Self-check tests for the from-scratch transformer.

Run from the assignment root:

    python -m pytest tests/ -q

All four must pass before you start training. They run on CPU in a few
seconds. Passing tests do not guarantee a correct model, but a failing
test guarantees an incorrect one; the causality test in particular
catches the classic masking bugs.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig  # noqa: E402

CFG = GPTConfig(block_size=32, vocab_size=65, n_layer=2, n_head=2, n_embd=64, dropout=0.0)


def test_output_shapes():
    model = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (3, 17))
    logits, loss = model(x, x)
    assert logits.shape == (3, 17, CFG.vocab_size)
    assert loss.dim() == 0


def test_causality():
    """Changing a future token must not change past logits."""
    model = GPT(CFG).eval()
    x = torch.randint(0, CFG.vocab_size, (1, 16))
    with torch.no_grad():
        logits_a, _ = model(x)
        x_perturbed = x.clone()
        x_perturbed[0, 10] = (x_perturbed[0, 10] + 1) % CFG.vocab_size
        logits_b, _ = model(x_perturbed)
    assert torch.allclose(logits_a[0, :10], logits_b[0, :10], atol=1e-5), \
        "logits at positions < 10 changed when token 10 changed: attention is not causal"
    assert not torch.allclose(logits_a[0, 10:], logits_b[0, 10:], atol=1e-5), \
        "logits at positions >= 10 did not change at all: model may be ignoring input"


def test_init_loss_near_uniform():
    """At init, next-token loss on random data should be near ln(V).

    The targets must be the SHIFTED sequence, not the input itself:
    with tied embeddings the residual stream keeps a component along
    the current token's embedding, so a predict-the-same-token target
    scores below chance even at init. (Try it: pass targets=x and
    watch the loss drop to ~3.45. That is the weight tying talking.)
    """
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    x = torch.randint(0, CFG.vocab_size, (8, 33))
    with torch.no_grad():
        _, loss = model(x[:, :-1], x[:, 1:])
    expected = math.log(CFG.vocab_size)
    assert abs(loss.item() - expected) < 0.35, \
        f"init loss {loss.item():.3f} far from ln(V)={expected:.3f}: check init scale"


def test_can_overfit_tiny_batch():
    """200 steps on one batch should drive loss well below chance."""
    torch.manual_seed(0)
    model = GPT(CFG).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, CFG.vocab_size, (4, 32))
    for _ in range(200):
        _, loss = model(x, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 1.0, f"loss {loss.item():.3f} after 200 steps on one batch: not learning"
