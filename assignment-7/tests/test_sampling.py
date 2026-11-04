"""Part 2: the group sampler on a tiny random model (CPU)."""
import torch

import sft
from grpo import sample_group
from model import GPT, GPTConfig


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=64, vocab_size=50257, n_layer=1, n_head=2, n_embd=16, bias=True)).eval()


def test_shapes_and_stopping():
    m = tiny()
    prompt = sft.ENC.encode("name[Aromi]\nDescription:")
    comps = sample_group(m, prompt, 4, 1.0, 12, "cpu")
    assert len(comps) == 4
    for c in comps:
        assert 1 <= len(c) <= 12
        assert sft.EOT not in c[:-1], "nothing after the first EOT"


def test_cold_sampling_is_greedy_and_identical_across_the_group():
    m = tiny()
    prompt = sft.ENC.encode("name[Aromi]\nDescription:")
    comps = sample_group(m, prompt, 3, 1e-4, 8, "cpu")
    assert comps[0] == comps[1] == comps[2]
    ids = torch.tensor([prompt])
    greedy = []
    with torch.no_grad():
        for _ in range(8):
            logits, _ = m(ids)
            nxt = logits[0, -1].argmax().item()
            greedy.append(nxt)
            if nxt == sft.EOT:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    assert comps[0] == greedy
