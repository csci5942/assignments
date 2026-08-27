"""Decoder-only transformer, written from scratch for CSCI 5942 A1.

Every module here is small enough to read in one sitting, and that is
the point. After this assignment you should be able to close the file
and rewrite it from memory. No HuggingFace, flash-attn import, or
config dataclass with forty fields.

"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 256   # maximum context length T
    vocab_size: int = 65    # char-level Shakespeare
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Input  x: (B, T, C)  batch, time, channels (C = n_embd)
    Output y: (B, T, C)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")


class MLP(nn.Module):
    """Position-wise feed-forward: expand 4x, nonlinearity, project back."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the output classifier reuses the token embedding.
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        # GPT-2 style scaled init on residual projections.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count N used in the scaling-law fits.

        With weight tying, wte IS lm_head, so subtracting wpe alone
        gives the standard "non-embedding" count.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.block_size
        pos = torch.arange(T, device=idx.device)
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        """Autoregressive sampling, one token per forward pass."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
