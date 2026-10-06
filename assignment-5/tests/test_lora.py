"""Part 4: LoRALinear, apply_lora, merge_lora (CPU, a tiny random GPT)."""
import torch

import lora
from model import GPT, GPTConfig

CFG = GPTConfig(block_size=32, vocab_size=97, n_layer=2, n_head=2, n_embd=16, dropout=0.0, bias=True)


def tiny():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    x = torch.randint(0, CFG.vocab_size, (3, 20))
    return model, x


def test_adapted_model_starts_as_the_base_model():
    model, x = tiny()
    with torch.no_grad():
        before, _ = model(x)
    lora.apply_lora(model, r=2, alpha=4.0)
    with torch.no_grad():
        after, _ = model(x)
    assert torch.allclose(before, after), "B is zero at init, so the output must not move"


def test_only_adapters_train_and_the_count_is_right():
    model, _ = tiny()
    n = lora.apply_lora(model, r=2, alpha=4.0)
    names = [k for k, p in model.named_parameters() if p.requires_grad]
    assert names and all(k.endswith(".A") or k.endswith(".B") for k in names)
    # per block: c_attn (16 -> 48), attn c_proj (16 -> 16), c_fc (16 -> 64), mlp c_proj (64 -> 16)
    per_block = 2 * ((16 + 48) + (16 + 16) + (16 + 64) + (64 + 16))
    assert n == CFG.n_layer * per_block


def test_merge_reproduces_the_adapted_model():
    model, x = tiny()
    lora.apply_lora(model, r=2, alpha=4.0)
    with torch.no_grad():                      # move B so the adapter does something
        for k, p in model.named_parameters():
            if k.endswith(".B"):
                p.normal_(0, 0.5)
        adapted, _ = model(x)
    lora.merge_lora(model)
    assert not any(isinstance(m, lora.LoRALinear) for m in model.modules())
    with torch.no_grad():
        merged, _ = model(x)
    assert torch.allclose(adapted, merged, atol=1e-4)
    assert all(p.requires_grad for p in model.parameters())
