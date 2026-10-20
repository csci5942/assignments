"""A1's decoder-only transformer, plus a KV cache for generation.

The model is your A1 solution, unchanged except that `forward` takes an
optional `cache`. `from_pretrained()` at the bottom loads OpenAI's GPT-2
weights into it (as in A5).
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


class KVCache:
    """Keys and values for every layer, allocated once.

    k[i], v[i]: (B, n_head, max_len, head_dim) for layer i. Slots
    0 .. pos-1 are filled. `pos` is shared by every row: each forward pass
    writes the same slots in every row.

    Prompts of different lengths are padded on the left
    (see left_pad), so every row's last prompt token is in the same slot
    and the next token goes into the same slot `pos` for all rows.
    pad: (B,) long on the cache's device, or None for no padding. Row b's
    first pad[b] slots hold pad tokens, and its real tokens start at slot
    pad[b]. Prompts "A B" and "C D E F" give pad = [2, 0]:

        slot        0  1  2  3 | 4  5   (decode)
        row 0       -  -  A  B | .  .
        row 1       C  D  E  F | .  .
        position 0  0  0  0  1 | 2  3
        position 1  0  1  2  3 | 4  5
    """

    def __init__(self, config: GPTConfig, batch_size: int, max_len: int,
                 pad: torch.Tensor | None = None, device=None, dtype=None):
        """Implement: allocate self.k and self.v, lists with one tensor per layer,
        to use as your KV cache memory. Allocate once
        and fill with zeros (do not grow the cache tensor)."""
        assert max_len <= config.block_size
        self.max_len = max_len
        self.pad = pad
        self.pos = 0
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    def begin(self, T: int) -> torch.Tensor:
        """Called once per forward pass, before the embeddings.

        Compute the new tokens' position ids, based on the padding for each
        sequence in the batch: a token's position is its slot minus its row's
        pad, clamped at 0.

        Returns: positions, a long tensor on the cache's device.
          - pad is None: shape (T,), the slot numbers pos .. pos+T-1.
          - otherwise: shape (B, T), row b's slot numbers minus pad[b],
            clamped at 0.
        In the class docstring's example, begin(4) on an empty cache
        returns [[0, 0, 0, 1], [0, 1, 2, 3]], and the next begin(1)
        returns [[2], [4]].
        """
        if self.pos + T > self.max_len:
            raise ValueError(f"cache full: {self.pos} + {T} > max_len {self.max_len}")
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Called in every layer with the new tokens' keys and values, k and
        v of shape (B, n_head, T, head_dim).

        Implement:
          1. Update your KV cache with the newly-computed keys and values.
          2. Build the attention mask for the T new tokens over all S = pos+T
             filled slots. New token t sits in slot pos+t and may attend to
             slot s when
               - s <= pos+t (causal), and
               - s >= pad[b], with padding: no query sees a pad slot.

        Returns: (K, V, mask).
          - K, V: slots 0 .. S-1 of self.k[layer] and self.v[layer], shape
            (B, n_head, S, head_dim). S is the number of filled cache slots
            once the new tokens are written.
          - mask: bool, True where attention is allowed, with dimensions
            (B, 1, T, S).
        """
        T = k.size(2)
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    def end(self, T: int):
        """Finish the forward pass: the T new tokens are now cached."""
        self.pos += T

    def stats(self) -> dict:
        """Token slots reserved vs holding a real token."""
        B = self.k[0].size(0)
        pad = 0 if self.pad is None else int(self.pad.sum())
        return {"reserved": B * self.max_len, "used": B * self.pos - pad}


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Input  x: (B, T, C)  batch, time, channels (C = n_embd)
    Output y: (B, T, C)

    With a cache, x holds only the T new tokens; the cache stores their
    keys and values and returns everything they may attend to.
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

    def forward(self, x: torch.Tensor, cache: KVCache | None = None, layer: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)                                  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)               # 3 x (B, T, C)
        hd = C // self.n_head
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        drop = self.dropout if self.training else 0.0
        if cache is None:
            # Fused causal attention, as in A4 and A5: no (B, nh, T, T) tensor.
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        else:
            k, v, mask = cache.update(layer, k, v)            # all S cached slots
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
                                                              # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)      # re-assemble heads
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward: expand 4x, nonlinearity, project back."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None, layer: int = 0) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cache, layer)
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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                cache: KVCache | None = None):
        """idx: (B, T) token ids. Without a cache, the whole sequence. With
        one, only the T new tokens; the cache supplies their positions and
        everything they attend to."""
        B, T = idx.shape
        if cache is None:
            assert T <= self.config.block_size
            pos = torch.arange(T, device=idx.device)                # (T,)
        else:
            pos = cache.begin(T)                                    # (T,) or (B, T)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)   # (B, T, C)
        x = self.transformer.drop(x)
        for i, block in enumerate(self.transformer.h):
            x = block(x, cache, i)
        if cache is not None:
            cache.end(T)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                    # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, use_cache: bool = False,
                 pad: torch.Tensor | None = None):
        """Autoregressive sampling. temperature=0 is greedy.

        Without a cache: one forward pass over the whole sequence per new
        token (A1). With one: a forward pass over the prompt (prefill),
        then one over each new token alone (decode). The cache lives on the
        model's device and dtype and holds prompt + max_new_tokens slots,
        which must fit in block_size. pad (B,) gives each row's left
        padding; it needs use_cache.
        """
        self.eval()
        if not use_cache:
            assert pad is None, "padding needs the cache"
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                idx_next = sample(logits[:, -1, :], temperature, top_k)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx
        B, T = idx.shape
        p = next(self.parameters())
        cache = KVCache(self.config, B, T + max_new_tokens, pad=pad, device=p.device, dtype=p.dtype)
        logits, _ = self(idx, cache=cache)                 # prefill: the whole prompt at once
        for i in range(max_new_tokens):
            idx_next = sample(logits[:, -1, :], temperature, top_k)
            idx = torch.cat((idx, idx_next), dim=1)
            if i < max_new_tokens - 1:                     # decode: only the new token
                logits, _ = self(idx_next, cache=cache)
        return idx


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None):
    """(B, V) logits -> (B, 1) next tokens. temperature=0 is greedy."""
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def left_pad(prompts: list[list[int]], pad_id: int = 0):
    """Batch prompts of different lengths, padding on the left.
    Returns idx (B, T_max), with pad_id in front of the shorter prompts,
    and pad (B,), the number of pad tokens in each row. Both are on the
    CPU; move them to the model's device."""
    T = max(len(p) for p in prompts)
    idx = torch.tensor([[pad_id] * (T - len(p)) + list(p) for p in prompts])
    pad = torch.tensor([T - len(p) for p in prompts])
    return idx, pad


# ---------------------------------------------------------------------
# GPT-2 weights (given, from A5)
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


def gpt2_config(name: str) -> GPTConfig:
    n_layer, n_head, n_embd = GPT2_SHAPES[name]
    return GPTConfig(block_size=1024, vocab_size=50257, n_layer=n_layer,
                     n_head=n_head, n_embd=n_embd, dropout=0.0, bias=True)


def from_pretrained(name: str = "gpt2") -> GPT:
    """Build GPT-2 `name` and load OpenAI's weights (openai-community/<name>
    on the Hugging Face Hub; cached under ~/.cache/huggingface)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    model = GPT(gpt2_config(name))
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
    ours["lm_head.weight"] = ours["transformer.wte.weight"]   # tied, as in GPT-2
    model.load_state_dict(ours)
    return model.eval()
