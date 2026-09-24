"""Part 2: does your tensor parallel model compute what one GPU computes?

    python -m pytest tests/test_tp.py -q

Four gloo ranks on a CPU, as in Part 1. No GPU, no corpus, no allocation.

Tensor parallelism has a property Part 1's data parallelism does not: a
sharded model and the dense model it came from are *the same model*, not
merely two models that should converge alike. `parallelize_gpt` takes
each rank's slice of the weights the dense model already holds, so every
test here compares against an exact answer rather than a tolerance on a
loss curve.

The failure that matters is the quiet one. A tensor parallel model with
a missing collective still runs, still produces a loss of roughly the
right size, and is wrong -- so these tests compare tensors.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from trainer.model import GPT, GPTConfig  # noqa: E402

from conftest import run_distributed  # noqa: E402

VOCAB = 512
SEED = 1337


def config(dropout: float = 0.0, vocab: int = VOCAB) -> GPTConfig:
    return GPTConfig(block_size=32, vocab_size=vocab, n_layer=2, n_head=4,
                     n_embd=64, dropout=dropout)


def dense_model(dropout: float = 0.0, vocab: int = VOCAB) -> GPT:
    torch.manual_seed(SEED)
    return GPT(config(dropout, vocab))


def close(got, want, what, rank, atol=2e-5, rtol=1e-4):
    diff = (got - want).abs().max().item()
    scale = want.abs().max().item()
    if diff <= atol + rtol * scale:
        return
    raise AssertionError(
        f"rank {rank}: {what} does not match the single-GPU result.\n"
        f"  max abs difference: {diff:.3e}  (reference max |x| = "
        f"{scale:.3e})\n"
        f"  Tensor parallelism is an exact rearrangement, not an "
        f"approximation: the sharded model holds slices of the very same "
        f"weights, so anything above floating-point noise is a missing or "
        f"misplaced collective.")


# ---------------------------------------------------------------------
# the two linears
# ---------------------------------------------------------------------
def _linear_body(rank, world, kind):
    from trainer.tp import (ColumnParallelLinear, RowParallelLinear,
                            _shard_cols, _shard_rows)
    from trainer.mesh import Mesh

    mesh = Mesh(tp=world, rank=rank, world_size=world)
    torch.manual_seed(SEED)
    dense = torch.nn.Linear(64, 128, bias=False)
    x = torch.randn(3, 5, 64, requires_grad=True)
    want = dense(x)

    gx_want = torch.autograd.grad(want.sum(), x, retain_graph=True)[0]
    per = 64 // world

    if kind == "column":
        # Forward: gather the shards so the full (.., 128) can be compared.
        # `gather_output` is an inspection convenience and is not
        # differentiable, so the gradient is checked separately, below.
        layer = ColumnParallelLinear(64, 128, group=mesh.tp_group,
                                     gather_output=True)
        with torch.no_grad():
            layer.weight.copy_(_shard_rows(dense.weight, mesh.tp_group))
        close(layer(x.detach()), want, "column-parallel forward", rank)

        # Backward through `f`: each rank differentiates only its own
        # slice of W, so the gradient w.r.t. the shared input is correct
        # only once those partial gradients are summed. This is the test
        # that fails if `f` is an identity in both directions.
        layer.gather_output = False
        x2 = x.detach().clone().requires_grad_(True)
        torch.autograd.grad(layer(x2).sum(), x2)[0]
        close(torch.autograd.grad(layer(x2).sum(), x2)[0], gx_want,
              "column-parallel input gradient", rank)
    else:
        layer = RowParallelLinear(64, 128, group=mesh.tp_group)
        with torch.no_grad():
            layer.weight.copy_(_shard_cols(dense.weight, mesh.tp_group))
        got = layer(x[..., rank * per:(rank + 1) * per])
        close(got, want, "row-parallel forward", rank)
        gx_got = torch.autograd.grad(got.sum(), x)[0]
        close(gx_got[..., rank * per:(rank + 1) * per],
              gx_want[..., rank * per:(rank + 1) * per],
              "row-parallel input gradient", rank)


@pytest.mark.parametrize("kind", ["column", "row"])
def test_parallel_linear_matches_dense(kind):
    run_distributed(_linear_body, 4, kind)


# ---------------------------------------------------------------------
# the MLP, and the given attention
# ---------------------------------------------------------------------
def _module_body(rank, world, which):
    from trainer.tp import (TensorParallelAttention, TensorParallelMLP,
                            _shard_cols, _shard_qkv, _shard_rows)
    from trainer.mesh import Mesh

    mesh = Mesh(tp=world, rank=rank, world_size=world)
    cfg = config()
    dense = dense_model()
    block = dense.transformer.h[0]
    x = torch.randn(2, cfg.block_size, cfg.n_embd)

    if which == "mlp":
        want = block.mlp(x)
        par = TensorParallelMLP(cfg, group=mesh.tp_group)
        with torch.no_grad():
            par.c_fc.weight.copy_(
                _shard_rows(block.mlp.c_fc.weight, mesh.tp_group))
            par.c_proj.weight.copy_(
                _shard_cols(block.mlp.c_proj.weight, mesh.tp_group))
    else:
        want = block.attn(x)
        par = TensorParallelAttention(cfg, group=mesh.tp_group)
        with torch.no_grad():
            par.c_attn.weight.copy_(
                _shard_qkv(block.attn.c_attn.weight, cfg.n_head,
                           mesh.tp_group))
            par.c_proj.weight.copy_(
                _shard_cols(block.attn.c_proj.weight, mesh.tp_group))

    par.eval()
    block.eval()
    close(par(x), want, f"tensor parallel {which}", rank)


def test_mlp_matches_dense():
    run_distributed(_module_body, 4, "mlp")


def test_attention():
    """The given attention. If this fails the handout is broken, not you."""
    run_distributed(_module_body, 4, "attn")


def _head_divisibility_body(rank, world):
    """n_head must divide tp, and the error must say so."""
    from trainer.tp import TensorParallelAttention
    from trainer.mesh import Mesh

    mesh = Mesh(tp=world, rank=rank, world_size=world)
    bad = GPTConfig(block_size=32, vocab_size=VOCAB, n_layer=1, n_head=3,
                    n_embd=48, dropout=0.0)
    with pytest.raises(ValueError, match="n_head"):
        TensorParallelAttention(bad, group=mesh.tp_group)


def test_uneven_head_split_is_refused():
    run_distributed(_head_divisibility_body, 4)


# ---------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------
def _loss_body(rank, world):
    from trainer.tp import vocab_parallel_cross_entropy
    from trainer.mesh import Mesh

    mesh = Mesh(tp=world, rank=rank, world_size=world)
    torch.manual_seed(SEED)
    B, T, V = 3, 7, VOCAB
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))

    want = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1))

    per = V // world
    shard = logits[..., rank * per:(rank + 1) * per].detach().clone()
    shard.requires_grad_(True)
    got = vocab_parallel_cross_entropy(shard, targets, mesh.tp_group,
                                       rank * per)
    close(got, want, "vocabulary-parallel cross-entropy", rank)

    gw = torch.autograd.grad(want, logits)[0]
    gg = torch.autograd.grad(got, shard)[0]
    close(gg, gw[..., rank * per:(rank + 1) * per],
          "vocabulary-parallel cross-entropy gradient", rank)


def test_vocab_parallel_cross_entropy_matches_dense():
    run_distributed(_loss_body, 4)


def _loss_never_gathers_body(rank, world):
    """The point of the exercise: no (B, T, V) tensor is ever formed.

    Watches every all-reduce and asserts each one is (B, T)-shaped, so an
    implementation that all-gathers the logits and calls F.cross_entropy
    fails here even though it gets the right answer.
    """
    import torch.distributed as dist
    from trainer.tp import vocab_parallel_cross_entropy
    from trainer.mesh import Mesh

    mesh = Mesh(tp=world, rank=rank, world_size=world)
    B, T, V = 2, 5, VOCAB
    per = V // world
    shard = torch.randn(B, T, per, requires_grad=True)
    targets = torch.randint(0, V, (B, T))

    seen = []
    real_reduce, real_gather = dist.all_reduce, dist.all_gather

    def spy_reduce(t, *a, **k):
        seen.append(("all_reduce", t.numel()))
        return real_reduce(t, *a, **k)

    def spy_gather(out, t, *a, **k):
        seen.append(("all_gather", t.numel()))
        return real_gather(out, t, *a, **k)

    dist.all_reduce, dist.all_gather = spy_reduce, spy_gather
    try:
        vocab_parallel_cross_entropy(shard, targets, mesh.tp_group, rank * per)
    finally:
        dist.all_reduce, dist.all_gather = real_reduce, real_gather

    biggest = max((n for _, n in seen), default=0)
    assert biggest <= B * T, (
        f"rank {rank}: the largest collective moved {biggest} elements, but "
        f"nothing larger than B*T = {B * T} should cross. It looks like the "
        f"logits themselves were communicated. The whole point of the "
        f"vocabulary-parallel loss is that only (B, T) scalars move: three "
        f"all-reduces of {B * T} elements against {B * T * V} for the "
        f"all-gather.\n  collectives seen: {seen}")
    assert len(seen) >= 3, (
        f"rank {rank}: only {len(seen)} collectives; expected at least "
        f"three (MAX for the global maximum, SUM for the exponentials, "
        f"SUM for the target logit). Seen: {seen}")


def test_loss_does_not_move_the_logits():
    run_distributed(_loss_never_gathers_body, 4)


# ---------------------------------------------------------------------
# the whole model
# ---------------------------------------------------------------------
def _full_body(rank, world):
    from trainer.tp import TensorParallelGPT
    from trainer.mesh import Mesh

    mesh = Mesh(tp=world, rank=rank, world_size=world)
    cfg = config()
    dense = dense_model().eval()
    torch.manual_seed(SEED)
    idx = torch.randint(0, VOCAB, (2, cfg.block_size))
    targets = torch.randint(0, VOCAB, (2, cfg.block_size))
    want_logits, want_loss = dense(idx, targets)

    par = TensorParallelGPT(dense_model(), group=mesh.tp_group).eval()
    got_logits, got_loss = par(idx, targets)

    close(got_loss, want_loss, "full tensor parallel loss", rank)

    per = cfg.vocab_size // world
    close(got_logits, want_logits[..., rank * per:(rank + 1) * per],
          "full tensor parallel logits", rank)


def test_full_model_matches_one_gpu():
    run_distributed(_full_body, 4)


def _backward_body(rank, world):
    """Gradients too, which is what catches a wrong `f`/`f_bar` pairing."""
    from trainer.tp import TensorParallelGPT
    from trainer.mesh import Mesh

    mesh = Mesh(tp=world, rank=rank, world_size=world)
    cfg = config()
    torch.manual_seed(SEED)
    idx = torch.randint(0, VOCAB, (2, cfg.block_size))
    targets = torch.randint(0, VOCAB, (2, cfg.block_size))

    dense = dense_model().eval()
    _, loss = dense(idx, targets)
    loss.backward()

    par = TensorParallelGPT(dense_model(), group=mesh.tp_group).eval()
    _, ploss = par(idx, targets)
    ploss.backward()

    # The embeddings are replicated, so their gradients must agree
    # exactly -- and they only do if `f` summed across ranks.
    close(par.module.transformer.wte.weight.grad,
          dense.transformer.wte.weight.grad,
          "token embedding gradient", rank, atol=5e-5)
    close(par.module.transformer.ln_f.weight.grad,
          dense.transformer.ln_f.weight.grad,
          "final layernorm gradient", rank, atol=5e-5)


def test_backward_matches_one_gpu():
    run_distributed(_backward_body, 4)
