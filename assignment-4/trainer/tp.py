"""Tensor parallel training.

Four things to implement:

    f, f_bar                      the conjugate collective pair
    ColumnParallelLinear          split W by output column
    RowParallelLinear             split W by input row
    TensorParallelMLP             the pair, stacked

Everything here takes `group=` and shards across `tp` ranks. As in
Part 1, leaving it at the default is the same thing as the tensor
parallel group only while `tp` equals the world size.

To run the correctness suite:

    python -m pytest tests/test_tp.py -q
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .ddp import group_world_size


def tp_rank(group=None) -> int:
    """This process's index within `group`, 0-based."""
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    return dist.get_rank(group=group)


def split_size(total: int, group=None) -> int:
    """`total // tp`, error if uneven
    """
    world = group_world_size(group)
    if total % world:
        raise ValueError(
            f"cannot split {total} across {world} tensor parallel ranks; "
            f"tp must divide it exactly")
    return total // world


# ---------------------------------------------------------------------
# 1. the conjugate pair
# ---------------------------------------------------------------------
# Every tensor parallel region is bracketed by two operators that are
# each other's mirror image. `f` does nothing on the way in and sums
# gradients on the way out; `f_bar` sums activations on the way out and
# does nothing on the way back.


class _F(torch.autograd.Function):
    """Identity forward, all-reduce backward. Used at the *start* of a
    tensor parallel region.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous() # in-place all-reduce requires contiguous memory

        # All-reduce (sum) the gradient across the tensor parallel group.
        raise NotImplementedError("Part 2: _F.backward")


class _FBar(torch.autograd.Function):
    """All-reduce forward, identity backward. Used at the *end* of a
    tensor parallel region.
    """

    @staticmethod
    def forward(ctx, x, group):
        x = x.contiguous() # in-place all-reduce requires contiguous memory
        raise NotImplementedError("Part 2: _FBar.forward")

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def f(x: torch.Tensor, group=None) -> torch.Tensor:
    """Identity forward, all-reduce backward."""
    if group_world_size(group) <= 1:
        return x
    return _F.apply(x, group)


def f_bar(x: torch.Tensor, group=None) -> torch.Tensor:
    """All-reduce forward, identity backward."""
    if group_world_size(group) <= 1:
        return x
    return _FBar.apply(x, group)


# ---------------------------------------------------------------------
# 2. the two linears
# ---------------------------------------------------------------------
class ColumnParallelLinear(nn.Module):
    """`Y = X W^T`, with `W` split by output column.

    Rank `i` holds `W_i` of shape `(out // tp, in)` and computes
    `Y_i = X W_i^T`. 

    Your job is to implement a linear layer, 
    where the weight matrix is split across multiple GPUs by its output columns.

    Hint: you will have to use f or f_bar. Use `F.linear` 
    for the matrix multiplication.

    Note that `gather_output=True` needs to be implemented as well,
    which will `all_gather` the shards back into a full `(..., out)` tensor. 
    This may not always be used (e.g., if the next layer is also sharded).
    """

    def __init__(self, in_features: int, out_features: int, group=None,
                 bias: bool = False, gather_output: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_per_rank = split_size(out_features, group)
        self.group = group
        self.gather_output = gather_output
        self.weight = nn.Parameter(
            torch.empty(self.out_per_rank, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_per_rank)) if bias \
            else None
        # Tagged so `clip_grad_norm` knows this parameter is a *shard*:
        # its sum of squares has to be added across the tensor parallel
        # group, where a replicated parameter must be counted once.
        self.weight.tensor_parallel = True
        if self.bias is not None:
            self.bias.tensor_parallel = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Part 2: ColumnParallelLinear.forward")


class RowParallelLinear(nn.Module):
    """`Y = X W^T`, with `W` split by input row.

    Rank `i` holds `W_i` of shape `(out, in // tp)` and receives an
    input already sharded on its last dimension, so `X_i W_i^T` is a
    *partial* result: the true output is the sum across ranks.

    Hint: you will have to use f or f_bar.

    The bias, if any, is added **after** the reduction. Adding it before
    would add it once per rank.
    """

    def __init__(self, in_features: int, out_features: int, group=None,
                 bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_per_rank = split_size(in_features, group)
        self.group = group
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_per_rank))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # The weight is a shard; the bias is not -- it is added after the
        # reduction, so every rank holds the same one.
        self.weight.tensor_parallel = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Part 2: RowParallelLinear.forward")


# ---------------------------------------------------------------------
# 3. the MLP
# ---------------------------------------------------------------------
class TensorParallelMLP(nn.Module):
    """
    Implement a tensor-parallel version of the MLP in `model.py`.
    `c_fc` is column-parallel and `c_proj` is row-parallel, so the
    whole block costs exactly **one** all-reduce, at the end.
    """

    def __init__(self, config, group=None):
        super().__init__()
        raise NotImplementedError("Part 2: TensorParallelMLP.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Part 2: TensorParallelMLP.forward")


# ---------------------------------------------------------------------
# 4. attention 
# ---------------------------------------------------------------------
class TensorParallelAttention(nn.Module):
    """`model.CausalSelfAttention`, sharded by head.
    """

    def __init__(self, config, group=None):
        super().__init__()
        world = group_world_size(group)
        if config.n_head % world:
            raise ValueError(
                f"n_head={config.n_head} is not divisible by tp={world}. "
                f"Attention shards by head, so tp must divide the head "
                f"count -- not just the embedding width.")
        self.group = group
        self.n_head = config.n_head
        self.n_head_local = config.n_head // world
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = ColumnParallelLinear(
            config.n_embd, 3 * config.n_embd, group=group, bias=config.bias)
        self.c_proj = RowParallelLinear(
            config.n_embd, config.n_embd, group=group, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.c_attn(x)                       # (B, T, 3*C/tp)
        local = self.n_head_local * self.head_dim
        q, k, v = qkv.split(local, dim=2)
        q = q.view(B, T, self.n_head_local, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head_local, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head_local, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
            is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, local)
        return self.resid_dropout(self.c_proj(y))


# ---------------------------------------------------------------------
# 5. the loss
# ---------------------------------------------------------------------
class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Cross-entropy over a vocabulary split across ranks.
    """

    @staticmethod
    def forward(ctx, logits, targets, group, vocab_start):
        V_local = logits.shape[-1]
        flat = logits.reshape(-1, V_local).float()
        tgt = targets.reshape(-1)

        z_max = flat.max(dim=-1).values
        if group_world_size(group) > 1:
            dist.all_reduce(z_max, op=dist.ReduceOp.MAX, group=group)

        exp = torch.exp(flat - z_max.unsqueeze(-1))
        sum_exp = exp.sum(dim=-1)
        if group_world_size(group) > 1:
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=group)

        mine = (tgt >= vocab_start) & (tgt < vocab_start + V_local)
        local_idx = torch.where(mine, tgt - vocab_start,
                                torch.zeros_like(tgt))
        z_y = torch.where(
            mine, flat.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1),
            torch.zeros_like(z_max))
        if group_world_size(group) > 1:
            dist.all_reduce(z_y, op=dist.ReduceOp.SUM, group=group)

        loss = torch.log(sum_exp) + z_max - z_y
        ctx.save_for_backward(exp, sum_exp, mine, local_idx)
        ctx.shape = logits.shape
        return loss.mean()

    @staticmethod
    def backward(ctx, grad_out):
        exp, sum_exp, mine, local_idx = ctx.saved_tensors
        # softmax over the *global* vocabulary, formed locally: the
        # numerator is this rank's and the denominator was reduced
        # in forward, so no collective is needed here at all.
        grad = exp / sum_exp.unsqueeze(-1)
        # ... minus one at the target, on whichever rank owns it.
        rows = torch.arange(grad.shape[0], device=grad.device)
        grad[rows[mine], local_idx[mine]] -= 1.0
        grad = grad * (grad_out / grad.shape[0])
        return grad.view(ctx.shape).to(exp.dtype), None, None, None


def vocab_parallel_cross_entropy(logits: torch.Tensor,
                                 targets: torch.Tensor,
                                 group=None,
                                 vocab_start: int | None = None):
    """Mean cross-entropy over a vocabulary-sharded logit tensor.

    `logits` is this rank's `(B, T, V/tp)` shard; `targets` is the full
    `(B, T)` of global token ids. Returns a scalar, matching
    `F.cross_entropy(logits.view(-1, V), targets.view(-1))` on one rank.
    """
    if vocab_start is None:
        vocab_start = tp_rank(group) * logits.shape[-1]
    return _VocabParallelCrossEntropy.apply(
        logits, targets, group, vocab_start)


# ---------------------------------------------------------------------
# 6. putting it together
# ---------------------------------------------------------------------
def _shard_rows(dense: torch.Tensor, group=None) -> torch.Tensor:
    """This rank's slice of a column-parallel weight `(out, in)`."""
    per = split_size(dense.shape[0], group)
    i = tp_rank(group)
    return dense[i * per:(i + 1) * per].clone()


def _shard_cols(dense: torch.Tensor, group=None) -> torch.Tensor:
    """This rank's slice of a row-parallel weight `(out, in)`."""
    per = split_size(dense.shape[1], group)
    i = tp_rank(group)
    return dense[:, i * per:(i + 1) * per].clone()


def _shard_qkv(dense: torch.Tensor, n_head: int, group=None) -> torch.Tensor:
    """This rank's heads from a fused `(3C, C)` qkv weight.

    See `TensorParallelAttention` for why this cannot be a contiguous
    slice of the `3C` rows.
    """
    three_c, c = dense.shape
    head_dim = c // n_head
    w = dense.view(3, n_head, head_dim, c)
    per = split_size(n_head, group)
    i = tp_rank(group)
    return w[:, i * per:(i + 1) * per].reshape(-1, c).clone()


@torch.no_grad()
def parallelize_gpt(model, group=None):
    """Convert a dense `GPT` in place, keeping its weights.
    """
    if group_world_size(group) <= 1:
        return model
    cfg = model.config
    for block in model.transformer.h:
        attn = TensorParallelAttention(cfg, group=group)
        attn.c_attn.weight.copy_(
            _shard_qkv(block.attn.c_attn.weight, cfg.n_head, group))
        attn.c_proj.weight.copy_(
            _shard_cols(block.attn.c_proj.weight, group))
        block.attn = attn

        mlp = TensorParallelMLP(cfg, group=group)
        mlp.c_fc.weight.copy_(_shard_rows(block.mlp.c_fc.weight, group))
        mlp.c_proj.weight.copy_(_shard_cols(block.mlp.c_proj.weight, group))
        block.mlp = mlp

    head = ColumnParallelLinear(cfg.n_embd, cfg.vocab_size, group=group)
    head.weight.copy_(_shard_rows(model.lm_head.weight, group))
    model.lm_head = head
    return model


class TensorParallelGPT(nn.Module):
    """A `GPT` whose blocks and output head are sharded.
    """

    def __init__(self, module, group=None):
        super().__init__()
        self.config = module.config
        self.module = parallelize_gpt(module, group)
        self.group = group
        self.vocab_start = tp_rank(group) * split_size(
            module.config.vocab_size, group) if group_world_size(group) > 1 \
            else 0

    def num_params(self, non_embedding: bool = False) -> int:
        return self.module.num_params(non_embedding=non_embedding)

    def param_bytes(self, optimizer_states: int = 2) -> dict:
        """Per-rank memory buckets, from the sharded parameter count."""
        return self.module.param_bytes(optimizer_states=optimizer_states)

    def forward(self, idx, targets=None):
        m = self.module
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = m.transformer.wte(idx) + m.transformer.wpe(pos)
        x = m.transformer.drop(x)
        for block in m.transformer.h:
            x = block(x)
        x = m.transformer.ln_f(x)
        logits = m.lm_head(x)                      # (B, T, V/tp)
        loss = None
        if targets is not None:
            loss = vocab_parallel_cross_entropy(
                logits, targets, self.group, self.vocab_start)
        return logits, loss
