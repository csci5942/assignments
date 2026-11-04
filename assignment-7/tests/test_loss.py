"""Part 2: the GRPO loss on hand-sized tensors (CPU, no model)."""
import math

import torch

from grpo import grpo_loss


def test_beta_zero_is_the_advantage_weighted_logprob():
    logp = torch.log(torch.tensor([[0.5, 0.25], [0.1, 1.0]]))
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])          # row 2 has one real token
    adv = torch.tensor([2.0, -1.0])
    loss, kl = grpo_loss(logp, logp.clone(), mask, adv, beta=0.0)
    row1 = -2.0 * (math.log(0.5) + math.log(0.25)) / 2
    row2 = 1.0 * math.log(0.1)
    assert math.isclose(loss.item(), (row1 + row2) / 2, rel_tol=1e-5)
    assert math.isclose(kl.item(), 0.0, abs_tol=1e-7)


def test_k3_is_zero_at_the_reference_and_positive_away_from_it():
    logp = torch.log(torch.tensor([[0.5, 0.5]]))
    mask = torch.ones(1, 2)
    adv = torch.zeros(1)
    loss, kl = grpo_loss(logp, logp.clone(), mask, adv, beta=1.0)
    assert math.isclose(loss.item(), 0.0, abs_tol=1e-7)
    ref = torch.log(torch.tensor([[0.9, 0.1]]))
    loss, kl = grpo_loss(logp, ref, mask, adv, beta=1.0)
    d = (ref - logp)
    expect = (torch.exp(d) - d - 1).mean().item()
    assert kl.item() > 0 and math.isclose(kl.item(), expect, rel_tol=1e-5)
    assert math.isclose(loss.item(), expect, rel_tol=1e-5)


def test_padding_carries_no_loss():
    logp = torch.log(torch.tensor([[0.5, 0.001]]))         # the second token is padding
    mask = torch.tensor([[1.0, 0.0]])
    adv = torch.tensor([1.0])
    loss, _ = grpo_loss(logp, logp.clone(), mask, adv, beta=0.0)
    assert math.isclose(loss.item(), -math.log(0.5), rel_tol=1e-5)
