"""Part 3: do both pipeline schedules compute what one GPU computes?

    python -m pytest tests/test_pp.py -q

Four gloo ranks on a CPU, as in Parts 1 and 2.

A pipeline stage holds the *same* weights the dense model holds -- the
stage is built from it, not initialised separately -- so every gradient
here is compared against an exact answer.

The failure that matters is silent. A 1F1B schedule that backwards its
micro-batches in the wrong order still runs, still produces a loss of
the right size, and computes a different gradient. So these tests
compare gradients, and they compare them per stage.
"""

import pytest

torch = pytest.importorskip("torch")

from trainer.model import GPT, GPTConfig  # noqa: E402

from conftest import run_distributed  # noqa: E402

VOCAB = 512
SEED = 1337


def config() -> GPTConfig:
    # 8 layers so that pp=4 gives two blocks a stage and pp=2 gives four.
    return GPTConfig(block_size=16, vocab_size=VOCAB, n_layer=8, n_head=4,
                     n_embd=32, dropout=0.0)


def dense_model() -> GPT:
    torch.manual_seed(SEED)
    return GPT(config())


def make_batches(m, micro, cfg):
    torch.manual_seed(99)
    return [(torch.randint(0, cfg.vocab_size, (micro, cfg.block_size)),
             torch.randint(0, cfg.vocab_size, (micro, cfg.block_size)))
            for _ in range(m)]


# ---------------------------------------------------------------------
# the splitter
# ---------------------------------------------------------------------
def test_split_is_contiguous_and_complete():
    from trainer.pp import split_layers
    for n_layer in (8, 12, 36, 7):
        for pp in (1, 2, 4):
            parts = split_layers(n_layer, pp)
            assert len(parts) == pp
            assert parts[0][0] == 0 and parts[-1][1] == n_layer
            for (_, a), (b, _) in zip(parts, parts[1:]):
                assert a == b, f"gap or overlap in {parts}"
            sizes = [e - s for s, e in parts]
            assert max(sizes) - min(sizes) <= 1, f"lopsided split {parts}"


# ---------------------------------------------------------------------
# the schedules
# ---------------------------------------------------------------------
def _schedule_body(rank, world, schedule, m, micro):
    """Both schedules must reproduce a single-GPU run's gradients."""
    from trainer.mesh import Mesh
    from trainer.pp import PipelineStage, pipeline_step

    mesh = Mesh(pp=world, rank=rank, world_size=world)
    cfg = config()
    batches = make_batches(m, micro, cfg)

    # -- the reference: one model, all micro-batches, plain accumulation
    ref = dense_model()
    ref.zero_grad(set_to_none=True)
    for x, y in batches:
        _, loss = ref(x, y)
        (loss / m).backward()

    # -- the pipeline
    model = dense_model()
    stage = PipelineStage(model, mesh)
    stage.zero_grad(set_to_none=True)
    loss, _compute = pipeline_step(stage, mesh, batches, schedule=schedule,
                         device="cpu", dtype=torch.float32, scale=m)

    # Every parameter this stage owns must match the dense model's.
    ref_named = dict(ref.named_parameters())
    checked = 0
    for name, p in stage.named_parameters():
        # stage parameter names are relative; map them back
        if name.startswith("h."):
            idx = int(name.split(".")[1])
            key = f"transformer.h.{stage.start + idx}." + \
                  ".".join(name.split(".")[2:])
        elif name.startswith(("wte.", "wpe.")):
            key = "transformer." + name
        elif name.startswith("ln_f."):
            key = "transformer." + name
        elif name.startswith("lm_head."):
            key = name
        else:
            raise AssertionError(f"unmapped stage parameter {name}")
        want = ref_named[key].grad
        assert p.grad is not None, (
            f"rank {rank}: {name} got no gradient at all under {schedule}. "
            f"A stage whose backward never ran is usually a schedule that "
            f"dropped a micro-batch.")
        diff = (p.grad - want).abs().max().item()
        scale_ = want.abs().max().item()
        assert diff <= 2e-5 + 1e-4 * scale_, (
            f"rank {rank}: {name} ({key}) disagrees with the single-GPU "
            f"gradient under {schedule}.\n"
            f"  max abs difference {diff:.3e}, reference max |g| "
            f"{scale_:.3e}\n"
            f"  Pipelining is an exact rearrangement, not an approximation. "
            f"The usual cause is backward running in the wrong order: the "
            f"stage in front sends gradients in the order it received "
            f"activations, so the queue must be drained from the front.")
        checked += 1
    assert checked, f"rank {rank}: stage owns no parameters"

    if mesh.is_last_stage:
        assert loss is not None and loss > 0


@pytest.mark.parametrize("schedule", ["afab", "1f1b"])
def test_schedule_matches_one_gpu(schedule):
    run_distributed(_schedule_body, 4, schedule, 8, 2)


@pytest.mark.parametrize("m", [4, 6, 9])
def test_1f1b_matches_across_micro_batch_counts(m):
    """m below, equal to and above the pipe depth."""
    run_distributed(_schedule_body, 4, "1f1b", m, 2)


def test_1f1b_at_pp2():
    run_distributed(_schedule_body, 2, "1f1b", 5, 2)


# ---------------------------------------------------------------------
# the two schedules against each other
# ---------------------------------------------------------------------
def _agree_body(rank, world, m, micro):
    """1F1B and AFAB must produce identical gradients, not merely close."""
    from trainer.mesh import Mesh
    from trainer.pp import PipelineStage, pipeline_step

    mesh = Mesh(pp=world, rank=rank, world_size=world)
    cfg = config()
    batches = make_batches(m, micro, cfg)

    grads = {}
    for schedule in ("afab", "1f1b"):
        stage = PipelineStage(dense_model(), mesh)
        stage.zero_grad(set_to_none=True)
        pipeline_step(stage, mesh, batches, schedule=schedule, device="cpu",
                      dtype=torch.float32, scale=m)
        grads[schedule] = {n: p.grad.clone()
                           for n, p in stage.named_parameters()}

    for name in grads["afab"]:
        a, b = grads["afab"][name], grads["1f1b"][name]
        diff = (a - b).abs().max().item()
        assert diff <= 2e-6 + 1e-5 * a.abs().max().item(), (
            f"rank {rank}: {name} differs between AFAB and 1F1B by "
            f"{diff:.3e}. The two schedules perform the same operations in "
            f"a different order; they should agree to floating-point noise "
            f"and nothing more.")


def test_schedules_agree_with_each_other():
    run_distributed(_agree_body, 4, 8, 2)


# ---------------------------------------------------------------------
# the property 1F1B exists for
# ---------------------------------------------------------------------
def _memory_body(rank, world, m):
    """1F1B holds at most `p` activation graphs; AFAB holds `m`.

    Counted rather than measured: a CPU has no allocator high-water mark
    to read, so the schedules are instrumented by counting how many
    forward results are alive at once.
    """
    from trainer.mesh import Mesh
    from trainer import pp as ppmod
    from trainer.pp import PipelineStage, pipeline_step

    mesh = Mesh(pp=world, rank=rank, world_size=world)
    cfg = config()
    batches = make_batches(m, 2, cfg)

    peaks = {}
    real_fwd, real_bwd = ppmod.forward_step, ppmod.backward_step
    for schedule in ("afab", "1f1b"):
        live = {"n": 0, "peak": 0}

        def fwd(*a, _l=live, **k):
            r = real_fwd(*a, **k)
            _l["n"] += 1
            _l["peak"] = max(_l["peak"], _l["n"])
            return r

        def bwd(*a, _l=live, **k):
            _l["n"] -= 1
            return real_bwd(*a, **k)

        ppmod.forward_step, ppmod.backward_step = fwd, bwd
        try:
            stage = PipelineStage(dense_model(), mesh)
            stage.zero_grad(set_to_none=True)
            pipeline_step(stage, mesh, batches, schedule=schedule,
                          device="cpu", dtype=torch.float32, scale=m)
        finally:
            ppmod.forward_step, ppmod.backward_step = real_fwd, real_bwd
        peaks[schedule] = live["peak"]

    assert peaks["afab"] == m, (
        f"rank {rank}: AFAB should hold all {m} micro-batches alive, "
        f"held {peaks['afab']}")
    assert peaks["1f1b"] <= world, (
        f"rank {rank}: 1F1B held {peaks['1f1b']} activation graphs alive "
        f"at once, but should never exceed the pipe depth p={world}. That "
        f"is the entire reason 1F1B exists -- if it holds `m` of them it "
        f"is AFAB with extra steps.")
    assert peaks["1f1b"] < peaks["afab"], (
        f"rank {rank}: 1F1B ({peaks['1f1b']}) did not hold fewer than "
        f"AFAB ({peaks['afab']})")


def test_1f1b_bounds_activation_memory():
    run_distributed(_memory_body, 4, 8)


# ---------------------------------------------------------------------
# tensor parallel and pipeline parallel, together
# ---------------------------------------------------------------------
def _tp_pp_body(rank, world, m, micro):
    """(tp=2, pp=2) must still compute what one GPU computes.

    Part 4 asks for `dp x tp x pp = 8` on eight GPUs, so more than one of
    the three is normally greater than one, and the placement experiment
    runs (2, 2, 2) specifically. Nothing else in the suite builds a stage
    out of a tensor-parallel model, and the two ways that goes wrong are
    both quiet:

    * `PipelineStage` reads `model.transformer`, which a
      `TensorParallelGPT` keeps one level down -- an AttributeError, so
      at least it is loud.
    * under tensor parallelism `lm_head` is column-parallel, so the last
      stage's logits are `(B, T, V/tp)`. Feeding those to a plain
      `cross_entropy` alongside targets drawn from the whole vocabulary
      computes a different number and reports it as the loss.
    """
    from trainer.mesh import Mesh
    from trainer.pp import PipelineStage, pipeline_step
    from trainer.tp import TensorParallelGPT

    mesh = Mesh(tp=2, pp=2, rank=rank, world_size=world)
    cfg = config()
    batches = make_batches(m, micro, cfg)

    # -- the reference: one dense model, plain accumulation
    ref = dense_model()
    ref_loss = 0.0
    for x, y in batches:
        _, loss = ref(x, y)
        ref_loss += loss.item() / m

    # -- the same computation, split two ways at once
    model = TensorParallelGPT(dense_model(), group=mesh.tp_group)
    stage = PipelineStage(model, mesh)
    assert list(stage.parameters()), (
        f"rank {rank}: the stage owns no parameters, so the optimizer "
        f"would get an empty list. Check that PipelineStage sliced the "
        f"blocks out of the tensor-parallel model rather than out of an "
        f"empty wrapper.")
    stage.zero_grad(set_to_none=True)
    loss, _compute = pipeline_step(stage, mesh, batches, schedule="1f1b",
                                   device="cpu", dtype=torch.float32,
                                   scale=m)

    if mesh.is_last_stage:
        assert loss is not None, f"rank {rank}: the last stage has no loss"
        got = float(loss)
        assert abs(got - ref_loss) <= 1e-4 + 1e-3 * abs(ref_loss), (
            f"rank {rank}: tp=2 x pp=2 computed loss {got:.6f} where one "
            f"GPU computes {ref_loss:.6f}.\n"
            f"  Splitting a model does not change what it computes. The "
            f"usual cause is the last stage using a plain cross_entropy "
            f"on column-parallel logits, which only ever sees this "
            f"rank's slice of the vocabulary.")

    grads = [p for p in stage.parameters() if p.grad is not None]
    assert grads, (
        f"rank {rank}: no parameter on this stage got a gradient, so the "
        f"backward pass never reached it.")


def test_tensor_parallel_and_pipeline_compose():
    run_distributed(_tp_pp_body, 4, 6, 2)


def _dp_tp_pp_body(rank, world, ddp_mode):
    """All three dimensions at once, wired the way `train.py` wires them.

    This is Part 4's own configuration -- `dp x tp x pp = 8` with none of
    them 1 -- and the placement experiment runs exactly it. The loss each
    rank reports is its *replica's*, over that replica's slice of the
    global batch, so the thing that must equal the single-process answer
    is the average across the data parallel group, not any one rank's
    number. Averaging gradients is what data parallelism does; the
    printed loss is not averaged for you.
    """
    import torch.distributed as dist

    from trainer.mesh import Mesh
    from trainer.model import GPT
    from trainer.pp import PipelineStage, pipeline_step
    from trainer.tp import TensorParallelGPT
    from trainer.train import wrap_model

    mesh = Mesh(dp=2, tp=2, pp=2, rank=rank, world_size=world)
    cfg = config()
    GLOBAL, MICRO = 8, 1

    torch.manual_seed(99)
    xs = torch.randint(0, cfg.vocab_size, (GLOBAL, cfg.block_size))
    ys = torch.randint(0, cfg.vocab_size, (GLOBAL, cfg.block_size))

    ref = dense_model()
    ref_loss = 0.0
    for i in range(GLOBAL // MICRO):
        _, l = ref(xs[i:i + 1], ys[i:i + 1])
        ref_loss += l.item() / (GLOBAL // MICRO)

    model = TensorParallelGPT(dense_model(), group=mesh.tp_group)
    stage = PipelineStage(model, mesh)
    _run, finish_grads, _ctx = wrap_model(stage, mesh, ddp_mode)

    per = GLOBAL // mesh.dp
    lo = mesh.dp_rank * per
    micro = [(xs[lo + i:lo + i + 1], ys[lo + i:lo + i + 1])
             for i in range(per)]
    loss, _compute = pipeline_step(stage, mesh, micro, schedule="1f1b",
                                   device="cpu", dtype=torch.float32,
                                   scale=per)
    finish_grads()

    assert any(p.grad is not None for p in stage.parameters()), (
        f"rank {rank}: nothing on this stage got a gradient with "
        f"dp=2 x tp=2 x pp=2 under --ddp {ddp_mode}.")

    if mesh.is_last_stage:
        # Average the replicas' losses back together over the dp group.
        t = torch.tensor([float(loss)])
        dist.all_reduce(t, group=mesh.dp_group)
        got = (t / mesh.dp).item()
        assert abs(got - ref_loss) <= 1e-4 + 1e-3 * abs(ref_loss), (
            f"rank {rank}: dp=2 x tp=2 x pp=2 under --ddp {ddp_mode} "
            f"averages to {got:.6f} where one process computes "
            f"{ref_loss:.6f}.\n"
            f"  Splitting a model three ways does not change what it "
            f"computes. Check that each replica took its own slice of "
            f"the global batch and no other.")


@pytest.mark.parametrize("ddp_mode", ["naive", "overlapped"])
def test_dp_tp_pp_compose(ddp_mode):
    run_distributed(_dp_tp_pp_body, 8, ddp_mode)
