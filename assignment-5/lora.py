"""Low-rank adaptation (Hu et al. 2021) for A4's model class. Part 4 edits
`LoRALinear.__init__`, `LoRALinear.forward` and `merge_lora`; the rest is
given.

A frozen weight W_0 of shape (d, k) gets a trainable rank-r update,

    h = W_0 x + (alpha / r) * B A x,    A: (r, k), B: (d, r),

A normal, B zero, so the adapted model equals the base model at step 0.
Only A and B carry gradients and optimizer state. At inference the update
folds into W_0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# The Linear layers in each block, by A1's names: c_attn (fused q, k, v),
# c_proj (attention output, and also the MLP down projection), c_fc (MLP
# up projection). Part 5 passes subsets.
ALL_TARGETS = ("c_attn", "c_proj", "c_fc")
ATTENTION_ONLY = ("c_attn",)          # q, k and v in one matrix
MLP_ONLY = ("c_fc",)                  # the expansion only; see Part 5


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable rank-r branch."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")


def apply_lora(model: nn.Module, r: int, alpha: float,
               targets=ALL_TARGETS) -> int:
    """Freeze `model`, wrap each Linear named in `targets` with a
    LoRALinear, return the trainable parameter count."""
    for p in model.parameters():
        p.requires_grad = False
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha))
    return count_trainable(model)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_state_dict(model: nn.Module) -> dict:
    """The adapter weights only."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith(".A") or k.endswith(".B")}


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Fold each adapter into its base weight and put the plain nn.Linear
    back. Logits must match the adapted model (tests/test_lora.py, 1e-4)."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                # --- YOUR IMPLEMENTATION HERE ---
                raise NotImplementedError("implement this block")
    for p in model.parameters():
        p.requires_grad = True
    return model
