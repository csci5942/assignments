"""Part 2: teacher-forced log-probabilities on a tiny random model (CPU)."""
import torch
import torch.nn.functional as F

import sft
from grpo import completion_logprobs
from model import GPT, GPTConfig


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=64, vocab_size=50257, n_layer=1, n_head=2, n_embd=16, bias=True)).eval()


def test_matches_the_models_own_cross_entropy():
    m = tiny()
    prompt = sft.ENC.encode("name[Aromi]\nDescription:")
    comp = sft.ENC.encode(" Aromi is a coffee shop.") + [sft.EOT]
    logp, mask = completion_logprobs(m, prompt, [comp], "cpu")
    assert logp.shape == mask.shape == (1, len(comp))
    assert mask.sum().item() == len(comp)
    ids = torch.tensor([prompt + comp])
    with torch.no_grad():
        logits, _ = m(ids)
    expect = -F.cross_entropy(logits[0, len(prompt) - 1:-1], ids[0, len(prompt):], reduction="none")
    assert torch.allclose(logp[0], expect, atol=1e-4)


def test_shorter_rows_are_masked_and_unchanged_by_padding():
    m = tiny()
    prompt = sft.ENC.encode("name[Aromi]\nDescription:")
    short = sft.ENC.encode(" Aromi.") + [sft.EOT]
    long = sft.ENC.encode(" Aromi is a coffee shop in the city centre.") + [sft.EOT]
    logp, mask = completion_logprobs(m, prompt, [short, long], "cpu")
    assert mask[0].tolist() == [1.0] * len(short) + [0.0] * (len(long) - len(short))
    alone, _ = completion_logprobs(m, prompt, [short], "cpu")
    assert torch.allclose(logp[0, :len(short)], alone[0], atol=1e-4)
