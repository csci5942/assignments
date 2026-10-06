"""A4's `trainer/model.py` (A1's transformer at GPT-2's shapes), unchanged,
plus `from_pretrained()` at the bottom, which loads OpenAI's released
GPT-2 weights. The loader sets two config values that differ from A4's:
`bias=True` (GPT-2 has biases) and `vocab_size=50257` (A4 pads to 50304;
a zero row in an untied `lm_head` would score logit 0). GPT-2 ties `wte`
and `lm_head`; this class does not, so the loader copies `wte` into
`lm_head`. GELU here is exact where GPT-2 used the tanh approximation;
`tests/test_pretrained.py` bounds the difference.

Not edited in Assignment 5.
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# GPT-2's BPE tokenizer emits 50257 distinct ids, but the vocabulary
# dimension here is 50304. The extra 47 rows are padding, and they exist
# because 50257 = 7 x 43 x 167 -- it is divisible by no power of two, so
# a vocabulary-parallel output head cannot shard it. 50304 is the next
# multiple of 128, which divides by every tensor parallel degree this
# assignment uses and is friendlier to the tensor cores besides.
#
# The padding rows are harmless: no token id in the corpus reaches them,
# so they are never a valid target and the softmax mass they receive is
# trained down to nothing. Every real implementation does this; Megatron
# calls it `make_vocab_size_divisible_by`.


@dataclass
class GPTConfig:
    block_size: int = 1024      # maximum context length T
    vocab_size: int = 50304     # GPT-2 BPE (50257), padded -- see below
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        """Build from a config JSON, ignoring the training-only keys."""
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def to_dict(self) -> dict:
        return asdict(self)


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
        # One fused projection for q, k, v; split on the channel dim.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)                                  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)               # 3 x (B, T, C)
        hd = C // self.n_head
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        # Fused causal attention: no (B, nh, T, T) tensor is ever written.
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
            is_causal=True)                                   # (B, nh, T, hd)

        y = y.transpose(1, 2).contiguous().view(B, T, C)      # re-assemble heads
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward: expand 4x, nonlinearity, project back.

    Part 2 shards this: `c_fc` column-parallel, `c_proj` row-parallel,
    one all-reduce after the second.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x)).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


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
        self.apply(self._init_weights)
        # GPT-2 style scaled init on residual projections.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -----------------------------------------------------------------
    # accounting
    # -----------------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wte.weight.numel()
            n -= self.transformer.wpe.weight.numel()
            n -= self.lm_head.weight.numel()
        return n

    def flops_per_token(self) -> float:
        """Forward + backward FLOPs for one token, for the MFU number.

        The 6N term is the usual "two FLOPs per parameter per token
        forward, four more backward". The second term is attention's
        quadratic part, which 6N does not include: each layer scores
        every query against every key and then mixes values, twice over
        in the backward pass.
        """
        cfg = self.config
        n = self.num_params(non_embedding=True) + \
            self.lm_head.weight.numel()          # the head does work too
        return 6.0 * n + 12.0 * cfg.n_layer * cfg.n_embd * cfg.block_size

    def param_bytes(self, optimizer_states: int = 2) -> dict:
        """The three memory buckets you can compute exactly, in bytes.
        """
        n = self.num_params()
        return {
            "params": 4 * n,
            "grads": 4 * n,
            "optimizer": 4 * n * optimizer_states,
        }

    # -----------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)   # (B, T, C)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                    # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss


# ---------------------------------------------------------------------
# Assignment 5: load the published GPT-2 weights (given)
# ---------------------------------------------------------------------

# (n_layer, n_head, n_embd) for the four released sizes.
GPT2_SHAPES = {
    "gpt2": (12, 12, 768),
    "gpt2-medium": (24, 16, 1024),
    "gpt2-large": (36, 20, 1280),
    "gpt2-xl": (48, 25, 1600),
}

# Hugging Face's checkpoint stores these four as (in, out) "Conv1D"
# weights; our nn.Linear wants (out, in).
_TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight",
               "mlp.c_fc.weight", "mlp.c_proj.weight")


def gpt2_config(name: str, dropout: float = 0.0) -> GPTConfig:
    n_layer, n_head, n_embd = GPT2_SHAPES[name]
    return GPTConfig(block_size=1024, vocab_size=50257, n_layer=n_layer,
                     n_head=n_head, n_embd=n_embd, dropout=dropout, bias=True)


def from_pretrained(name: str = "gpt2", dropout: float = 0.0) -> "GPT":
    """Build GPT-2 `name` and load OpenAI's weights (openai-community/<name>
    on the Hugging Face Hub; cached under ~/.cache/huggingface)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    model = GPT(gpt2_config(name, dropout))
    path = hf_hub_download(f"openai-community/{name}", "model.safetensors")
    theirs = load_file(path)
    ours = model.state_dict()
    for key, value in theirs.items():
        if key.endswith(".attn.bias") or key.endswith(".attn.masked_bias"):
            continue                        # their causal-mask buffer; we use is_causal
        if any(key.endswith(t) for t in _TRANSPOSED):
            value = value.t()
        target = key if key.startswith("transformer.") else "transformer." + key
        assert target in ours, f"unexpected key {key}"
        assert ours[target].shape == value.shape, \
            f"{target}: {tuple(ours[target].shape)} vs {tuple(value.shape)}"
        ours[target] = value.contiguous()
    ours["lm_head.weight"] = theirs["wte.weight"].clone()   # GPT-2 ties them
    model.load_state_dict(ours)
    return model
