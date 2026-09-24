"""Part 1: is your data parallel the same computation as one GPU?

    python -m pytest tests/test_ddp.py -q

Four ranks at micro-batch `B/4`, against one rank at batch `B`. No GPU,
no corpus, no allocation -- four forked processes talking over `gloo`.

The bar is not that your code runs. Data parallelism that is wrong in
any of the four usual ways still runs, still produces a loss that goes
down, and still finishes. These tests compare *gradients*, because the
loss curve is the one signal that will not tell you.

The four, in the order people hit them:

  1. summing gradients where you meant to average them
  2. averaging per-rank losses when the ranks saw different token counts
  3. letting each rank initialize its own weights
  4. letting each rank draw its own dropout mask

When a comparison fails, the message tries to name which one you have.
"""

import pytest

torch = pytest.importorskip("torch")

from trainer.data import Batcher, micro_batch_seed, open_corpus  # noqa: E402
from trainer.model import GPT, GPTConfig  # noqa: E402

from conftest import run_distributed  # noqa: E402

BLOCK = 64
GLOBAL_BATCH = 8
MICRO = 2
SEED = 1337
STEPS = 50


def config(dropout: float = 0.0) -> GPTConfig:
    return GPTConfig(block_size=BLOCK, vocab_size=512, n_layer=2, n_head=4,
                     n_embd=64, dropout=dropout)


def fresh_model(dropout: float = 0.0, seed: int = SEED) -> GPT:
    torch.manual_seed(seed)
    return GPT(config(dropout))


def named_grads(model) -> dict:
    return {n: (p.grad.clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}


# ---------------------------------------------------------------------
# the two computations being compared
# ---------------------------------------------------------------------
def single_rank_grads(model, batcher, step: int) -> dict:
    """One rank, the whole global batch, `GLOBAL_BATCH // MICRO` micro-steps.

    This is exactly `trainer/train.py`'s inner loop at world size 1, and
    it is the answer every configuration in this assignment has to
    reproduce.
    """
    accum = GLOBAL_BATCH // MICRO
    x_all, y_all = batcher.batch(step)
    model.zero_grad(set_to_none=True)
    for i in range(accum):
        torch.manual_seed(micro_batch_seed(SEED, step, i))
        x = x_all[i * MICRO:(i + 1) * MICRO]
        y = y_all[i * MICRO:(i + 1) * MICRO]
        _, loss = model(x, y)
        (loss / accum).backward()
    return named_grads(model)


def data_parallel_grads(model, batcher, step, rank, world, reduce_fn) -> dict:
    """This rank's slice, then whatever `reduce_fn` does to the gradients.

    `reduce_fn(model)` stands in for either `naive_all_reduce_grads` or
    `OverlappedDDP.finish_gradient_synchronization`.
    """
    local_batch = GLOBAL_BATCH // world
    accum = max(1, local_batch // MICRO)
    micro = local_batch // accum
    x_all, y_all = batcher.batch(step, dp_rank=rank, dp_world=world)
    model.zero_grad(set_to_none=True)
    for j in range(accum):
        # The *global* index of this micro-batch, so that the dropout
        # masks line up with the single-rank run's.
        torch.manual_seed(micro_batch_seed(SEED, step, rank * accum + j))
        x = x_all[j * micro:(j + 1) * micro]
        y = y_all[j * micro:(j + 1) * micro]
        _, loss = model(x, y)
        (loss / accum).backward()
    reduce_fn(model)
    return named_grads(model)


# ---------------------------------------------------------------------
# saying which bug it is
# ---------------------------------------------------------------------
def _ratio(got, want) -> float | None:
    """Median got/want over the entries where `want` is big enough to divide."""
    mask = want.abs() > want.abs().max() * 0.1
    if mask.sum() < 4:
        return None
    return (got[mask] / want[mask]).median().item()


def compare_grads(got, want, world, rank, atol=2e-6, rtol=1e-4) -> None:
    worst_name, worst_diff = None, 0.0
    for name, ref in want.items():
        mine = got[name]
        if ref is None or mine is None:
            assert ref is None and mine is None, \
                f"rank {rank}: {name}.grad is None on one side but not the other"
            continue
        diff = (mine - ref).abs().max().item()
        if diff > atol + rtol * ref.abs().max().item() and diff > worst_diff:
            worst_name, worst_diff = name, diff
    if worst_name is None:
        return

    mine, ref = got[worst_name], want[worst_name]
    r = _ratio(mine, ref)
    if r is not None and abs(r - world) < 0.05 * world:
        why = (f"Your gradients are {r:.2f}x the reference, and there are "
               f"{world} ranks. You summed where you meant to average "
               f"(bug 1). `dist.all_reduce` defaults to ReduceOp.SUM, and "
               f"each rank's gradient is already a mean over its own slice.")
    elif r is not None and abs(r - 1.0 / world) < 0.05:
        why = (f"Your gradients are {r:.3f}x the reference, which is "
               f"1/{world}. You divided twice -- once before the reduction "
               f"and once after, or once here and once in the loss.")
    elif r is not None and abs(r - 1.0) < 0.05:
        why = ("The scale is right but the values are not, so the ranks "
               "computed different things from the same data. Either the "
               "replicas were never made identical (bug 3 -- see "
               "broadcast_module_state) or something stochastic diverged "
               "(bug 4 -- see trainer.data.micro_batch_seed).")
    else:
        why = (f"Ratio to the reference is {r if r is None else round(r, 4)}. "
               f"Check that every rank all-reduces every gradient exactly "
               f"once, over the data parallel group.")

    raise AssertionError(
        f"rank {rank}: gradients do not match the single-rank reference.\n"
        f"  worst parameter: {worst_name}\n"
        f"  max abs difference: {worst_diff:.3e}  "
        f"(reference max |g| = {ref.abs().max().item():.3e})\n"
        f"  {why}")


# ---------------------------------------------------------------------
# step 0 gradients
# ---------------------------------------------------------------------
def _grads_body(rank, world, corpus_root, dropout, overlapped):
    from trainer.ddp import OverlappedDDP, naive_all_reduce_grads
    from trainer.mesh import Mesh

    mesh = Mesh(dp=world, rank=rank, world_size=world)
    batcher = Batcher(open_corpus(corpus_root, BLOCK), GLOBAL_BATCH, seed=SEED)

    want = single_rank_grads(fresh_model(dropout), batcher, step=0)

    model = fresh_model(dropout)
    if overlapped:
        ddp = OverlappedDDP(model, group=mesh.dp_group)
        got = data_parallel_grads(
            ddp, batcher, 0, rank, world,
            lambda m: m.finish_gradient_synchronization())
        got = {n.removeprefix("module."): g for n, g in got.items()}
    else:
        got = data_parallel_grads(
            model, batcher, 0, rank, world,
            lambda m: naive_all_reduce_grads(m, group=mesh.dp_group))

    compare_grads(got, want, world, rank)


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_naive_gradients_match_one_rank(corpus, dropout):
    """Four ranks at B/4 must produce one rank at B's gradient, exactly.

    The `dropout=0.1` case is the one `configs/verify.json` turns on.
    It passes only if every rank draws the mask that the single-rank run
    would have drawn for those same sequences, which is what
    `trainer.data.micro_batch_seed` is for.
    """
    run_distributed(_grads_body, 4, corpus, dropout, False)


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_overlapped_gradients_match_one_rank(corpus, dropout):
    run_distributed(_grads_body, 4, corpus, dropout, True)


# ---------------------------------------------------------------------
# bug 3, on its own
# ---------------------------------------------------------------------
def _broadcast_body(rank, world, _corpus_root):
    import torch.distributed as dist
    from trainer.ddp import broadcast_module_state
    from trainer.mesh import Mesh

    mesh = Mesh(dp=world, rank=rank, world_size=world)
    # Deliberately diverge: every rank initializes from its own seed.
    model = fresh_model(seed=SEED + rank)
    before = next(model.parameters()).clone()

    broadcast_module_state(model, group=mesh.dp_group)

    for name, p in model.named_parameters():
        gathered = [torch.empty_like(p) for _ in range(world)]
        dist.all_gather(gathered, p.data, group=mesh.dp_group)
        for other in range(world):
            assert torch.equal(gathered[0], gathered[other]), (
                f"after broadcast_module_state, rank {other}'s {name} still "
                f"differs from rank 0's. Averaging gradients keeps replicas "
                f"that already agree in agreement; it cannot make "
                f"disagreeing replicas agree, and nothing downstream will "
                f"tell you.")
    if rank != 0:
        assert not torch.equal(before, next(model.parameters())), \
            "the test did not actually diverge the replicas; it proves nothing"


def test_replicas_are_made_identical(corpus):
    run_distributed(_broadcast_body, 4, corpus)


def _ddp_broadcasts_body(rank, world, _corpus_root):
    import torch.distributed as dist
    from trainer.ddp import OverlappedDDP
    from trainer.mesh import Mesh

    mesh = Mesh(dp=world, rank=rank, world_size=world)
    ddp = OverlappedDDP(fresh_model(seed=SEED + rank), group=mesh.dp_group)
    for name, p in ddp.named_parameters():
        gathered = [torch.empty_like(p) for _ in range(world)]
        dist.all_gather(gathered, p.data, group=mesh.dp_group)
        assert all(torch.equal(gathered[0], g) for g in gathered), (
            f"OverlappedDDP did not synchronize {name} at construction. "
            f"Wrapping the model is the one moment you know every rank is "
            f"about to start from the same place; take it.")


def test_wrapping_synchronizes_the_replicas(corpus):
    run_distributed(_ddp_broadcasts_body, 4, corpus)


# ---------------------------------------------------------------------
# the group, not the world
# ---------------------------------------------------------------------
def _group_body(rank, world):
    """dp=2 x tp=2: reducing over the world instead of the dp group is wrong.

    In Part 1 the two are the same thing and this bug is invisible. Here
    they are not, which is the whole point of catching it now rather
    than in Part 4.
    """
    from trainer.ddp import naive_all_reduce_grads
    from trainer.mesh import Mesh

    mesh = Mesh(dp=2, tp=2, rank=rank, world_size=world)
    model = fresh_model()
    for p in model.parameters():
        p.grad = torch.full_like(p, float(1 << rank))

    naive_all_reduce_grads(model, group=mesh.dp_group)

    partners = mesh.group_ranks("dp")
    expect = sum(1 << r for r in partners) / len(partners)
    got = next(model.parameters()).grad.flatten()[0].item()

    # Reducing over the world instead of the group shows up as the sum
    # over *everyone*, divided by whichever count the code used.
    world_sum = sum(1 << r for r in range(world))
    reduced_everyone = any(abs(got - world_sum / d) < 1e-6
                           for d in (len(partners), world))
    assert abs(got - expect) < 1e-6, (
        f"rank {rank}: averaging over the dp group {partners} should give "
        f"{expect}, got {got}."
        + (f" That is the sum over all {world} ranks: you reduced over the "
           f"default (world) group instead of `group`. In Part 1 the two "
           f"are the same set of ranks and this bug is invisible; here they "
           f"are not, which is why the test exists now rather than in "
           f"Part 4." if reduced_everyone else ""))


def test_reduces_over_the_dp_group_not_the_world():
    run_distributed(_group_body, 4)


# ---------------------------------------------------------------------
# overlap and no_sync are optimizations, not changes of meaning
# ---------------------------------------------------------------------
def _equivalence_body(rank, world, corpus_root):
    from trainer.ddp import OverlappedDDP, naive_all_reduce_grads
    from trainer.mesh import Mesh

    mesh = Mesh(dp=world, rank=rank, world_size=world)
    batcher = Batcher(open_corpus(corpus_root, BLOCK), GLOBAL_BATCH, seed=SEED)

    naive = fresh_model()
    want = data_parallel_grads(
        naive, batcher, 3, rank, world,
        lambda m: naive_all_reduce_grads(m, group=mesh.dp_group))

    ddp = OverlappedDDP(fresh_model(), group=mesh.dp_group)
    got = data_parallel_grads(
        ddp, batcher, 3, rank, world,
        lambda m: m.finish_gradient_synchronization())
    compare_grads({n.removeprefix("module."): g for n, g in got.items()},
                  want, world, rank)


def test_overlapped_agrees_with_naive(corpus):
    run_distributed(_equivalence_body, 4, corpus)


def _no_sync_body(rank, world, corpus_root):
    """Two micro-batches per rank, reduced once at the end, must equal
    the same two reduced one at a time."""
    import contextlib
    from trainer.ddp import OverlappedDDP
    from trainer.mesh import Mesh

    mesh = Mesh(dp=world, rank=rank, world_size=world)
    batcher = Batcher(open_corpus(corpus_root, BLOCK), GLOBAL_BATCH, seed=SEED)
    local = GLOBAL_BATCH // world           # 4 with world=2
    micro, accum = local // 2, 2

    def run(use_no_sync):
        ddp = OverlappedDDP(fresh_model(), group=mesh.dp_group)
        x_all, y_all = batcher.batch(1, dp_rank=rank, dp_world=world)
        ddp.zero_grad(set_to_none=True)
        for j in range(accum):
            torch.manual_seed(micro_batch_seed(SEED, 1, rank * accum + j))
            ctx = ddp.no_sync() if (use_no_sync and j < accum - 1) \
                else contextlib.nullcontext()
            with ctx:
                _, loss = ddp(x_all[j * micro:(j + 1) * micro],
                              y_all[j * micro:(j + 1) * micro])
                (loss / accum).backward()
            if not use_no_sync:
                # Without no_sync the reduction for micro-batch j must be
                # finished before micro-batch j+1's backward writes to the
                # same p.grad, or the two race.
                ddp.finish_gradient_synchronization()
        ddp.finish_gradient_synchronization()
        return named_grads(ddp)

    compare_grads(run(True), run(False), world, rank)


def test_no_sync_changes_nothing_but_the_traffic(corpus):
    run_distributed(_no_sync_body, 2, corpus)


# ---------------------------------------------------------------------
# and over many steps
# ---------------------------------------------------------------------
def _curve_body(rank, world, corpus_root, steps):
    from trainer.ddp import OverlappedDDP
    from trainer.mesh import Mesh

    mesh = Mesh(dp=world, rank=rank, world_size=world)
    batcher = Batcher(open_corpus(corpus_root, BLOCK), GLOBAL_BATCH, seed=SEED)

    ddp = OverlappedDDP(fresh_model(), group=mesh.dp_group)
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

    # Only one rank carries the single-rank reference. It is the full
    # global batch, so running it on all four would quadruple the work to
    # re-derive the same number -- and any rank that diverged changes the
    # all-reduced gradient, which this rank sees.
    ref = ref_opt = None
    if rank == 0:
        ref = fresh_model()
        ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-3)

    local = GLOBAL_BATCH // world
    accum = max(1, local // MICRO)
    micro = local // accum

    for step in range(steps):
        ref_loss = 0.0
        if rank == 0:
            ref_opt.zero_grad(set_to_none=True)
            n_ref = GLOBAL_BATCH // MICRO
            xr, yr = batcher.batch(step)
            for i in range(n_ref):
                torch.manual_seed(micro_batch_seed(SEED, step, i))
                _, l = ref(xr[i * MICRO:(i + 1) * MICRO],
                           yr[i * MICRO:(i + 1) * MICRO])
                (l / n_ref).backward()
                ref_loss += l.item() / n_ref
            torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
            ref_opt.step()

        opt.zero_grad(set_to_none=True)
        xd, yd = batcher.batch(step, dp_rank=rank, dp_world=world)
        for j in range(accum):
            torch.manual_seed(micro_batch_seed(SEED, step, rank * accum + j))
            _, l = ddp(xd[j * micro:(j + 1) * micro],
                       yd[j * micro:(j + 1) * micro])
            (l / accum).backward()
        ddp.finish_gradient_synchronization()
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        opt.step()

        if rank == 0:
            drift = max((a - b).abs().max().item()
                        for a, b in zip(ref.parameters(), ddp.parameters()))
            assert drift < 1e-4, (
                f"after {step + 1} steps the data parallel weights have "
                f"drifted {drift:.2e} from the single-rank run (reference "
                f"loss {ref_loss:.4f}). A gradient wrong by a constant "
                f"factor survives one step and separates over many: AdamW "
                f"normalizes the gradient but not the weight decay, so the "
                f"two runs decay at different effective rates.")


def test_weights_track_the_single_rank_run_for_fifty_steps(corpus):
    """The slow-burn version. Step 0 can match while step 50 does not."""
    run_distributed(_curve_body, 4, corpus, STEPS)
