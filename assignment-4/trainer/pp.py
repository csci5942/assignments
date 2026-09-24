"""Pipeline parallel training.

Each rank gets a contiguous slice of the *layers* and passes activations
down the line. 

Almost all of this file is given. 
You only need to implement the following:

    one_forward_one_backward      

Note that you are writing a schedule, not new tensor code.
Everything 1F1B needs already exists below, just called in a different order.

Run the correctness suite:

    python -m pytest tests/test_pp.py -q
"""

from __future__ import annotations

import time
from collections import deque

import torch
import torch.distributed as dist
import torch.nn as nn

from .mesh import Mesh
from .tp import vocab_parallel_cross_entropy


# ---------------------------------------------------------------------
# 1. cutting the model up
# ---------------------------------------------------------------------
def split_layers(n_layer: int, pp: int) -> list[tuple[int, int]]:
    """Contiguous `(start, end)` block ranges, one per stage.

    Blocks are handed out as evenly as possible, the remainder going to
    the earlier stages.
    """
    if pp > n_layer:
        raise ValueError(
            f"pp={pp} exceeds n_layer={n_layer}: pipeline degree "
            f"cannot exceed the number of transformer blocks, or "
            f"some stages get none at all.")
    per, extra = divmod(n_layer, pp)
    out, start = [], 0
    for i in range(pp):
        end = start + per + (1 if i < extra else 0)
        out.append((start, end))
        start = end
    return out


class PipelineStage(nn.Module):
    """This rank's slice of a `GPT`. 

    The first stage owns the embeddings, the last owns the final norm
    and the output head, everyone owns a contiguous run of blocks.

    Takes either a plain `GPT` or a `TensorParallelGPT`, so that the two
    dimensions compose.
    """

    def __init__(self, model, mesh: Mesh):
        super().__init__()
        self.mesh = mesh
        self.config = model.config
        # `TensorParallelGPT` is a wrapper: the sharded `GPT` is inside
        # it, and its blocks are already tensor-parallel, so slicing them
        # into stages needs nothing special. What *is* special is the
        # loss -- see `forward`.
        inner = getattr(model, "module", model)
        self.tp_group = getattr(model, "group", None)
        self.vocab_start = getattr(model, "vocab_start", 0)
        start, end = split_layers(model.config.n_layer, mesh.pp)[mesh.pp_rank]
        self.start, self.end = start, end
        self.h = nn.ModuleList(list(inner.transformer.h)[start:end])
        self.first = mesh.is_first_stage
        self.last = mesh.is_last_stage
        if self.first:
            self.wte = inner.transformer.wte
            self.wpe = inner.transformer.wpe
            self.drop = inner.transformer.drop
        if self.last:
            self.ln_f = inner.transformer.ln_f
            self.lm_head = inner.lm_head

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        """Token ids in on the first stage, hidden states everywhere else.

        Returns `(hidden, None)` on every stage but the last, and
        `(logits, loss)` on the last.
        """
        if self.first:
            pos = torch.arange(x.shape[1], device=x.device)
            x = self.drop(self.wte(x) + self.wpe(pos))
        for block in self.h:
            x = block(x)
        if not self.last:
            return x, None
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            if self.tp_group is not None:
                # Under tensor parallelism `lm_head` is column-parallel,
                # so `logits` is (B, T, V/tp) and holds this rank's slice
                # of the vocabulary. Plain `cross_entropy` would index it
                # with targets drawn from the *whole* vocabulary
                loss = vocab_parallel_cross_entropy(
                    logits, targets, self.tp_group, self.vocab_start)
            else:
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def activation_shape(self, micro_batch: int) -> tuple[int, int, int]:
        """What a neighbor will send us, so `recv` can allocate first."""
        return (micro_batch, self.config.block_size, self.config.n_embd)

    def num_params(self, non_embedding: bool = False) -> int:
        """This *stage's* parameter count, not the whole model's.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.first:
            n -= self.wte.weight.numel() + self.wpe.weight.numel()
        if non_embedding and self.last:
            n -= self.lm_head.weight.numel()
        return n

    def param_bytes(self, optimizer_states: int = 2) -> dict:
        """Per-stage memory buckets: four bytes each for the parameters,
        their gradients, and AdamW's two moments."""
        n = sum(p.numel() for p in self.parameters())
        return {"params": 4 * n, "grads": 4 * n,
                "optimizer": 4 * n * optimizer_states}


# ---------------------------------------------------------------------
# 2. talking to the neighbors
# ---------------------------------------------------------------------
class Pipe:
    """Point-to-point transfer between adjacent stages. 

    Named for the direction the *data* moves: activations go forward,
    gradients go backward.
    (`send_forward` on the last stage does nothing, and
    `recv_forward` on the first returns None)

    `send_forward_recv_backward` and `send_backward_recv_forward` post 
    both halves as one batched non-blocking transfer, which cannot deadlock.
    Use these operations for the steady state.
    """

    def __init__(self, mesh: Mesh, shape, dtype, device):
        self.mesh = mesh
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def _buffer(self):
        return torch.empty(self.shape, dtype=self.dtype, device=self.device)

    # -- one direction at a time --------------------------------------
    def send_forward(self, x) -> None:
        if self.mesh.is_last_stage or x is None:
            return
        dist.send(x.to(self.dtype).contiguous(), dst=self.mesh.next_stage)

    def recv_forward(self):
        if self.mesh.is_first_stage:
            return None
        buf = self._buffer()
        dist.recv(buf, src=self.mesh.prev_stage)
        return buf.requires_grad_(True)

    def send_backward(self, grad) -> None:
        if self.mesh.is_first_stage or grad is None:
            return
        dist.send(grad.to(self.dtype).contiguous(), dst=self.mesh.prev_stage)

    def recv_backward(self):
        if self.mesh.is_last_stage:
            return None
        buf = self._buffer()
        dist.recv(buf, src=self.mesh.next_stage)
        return buf

    # -- both directions at once, for the steady state ----------------
    def _batch(self, ops):
        if not ops:
            return
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    def send_forward_recv_backward(self, x):
        """Hand an activation on, take a gradient back. One transfer."""
        if self.mesh.is_last_stage:
            return None
        out = self._buffer()
        payload = x.to(self.dtype).contiguous()
        self._batch([
            dist.P2POp(dist.isend, payload, self.mesh.next_stage),
            dist.P2POp(dist.irecv, out, self.mesh.next_stage),
        ])
        return out

    def send_backward_recv_forward(self, grad):
        """Hand a gradient back, take the next activation. One transfer."""
        if self.mesh.is_first_stage:
            return None
        out = self._buffer()
        ops = [dist.P2POp(dist.irecv, out, self.mesh.prev_stage)]
        if grad is not None:
            payload = grad.to(self.dtype).contiguous()
            ops.insert(0, dist.P2POp(dist.isend, payload,
                                     self.mesh.prev_stage))
        self._batch(ops)
        return out.requires_grad_(True)


def broadcast_loss(loss, mesh: Mesh, device):
    """Put the last stage's loss on every rank, for logging.
    """
    if mesh.pp <= 1:
        return loss
    t = torch.tensor([float(loss) if loss is not None else 0.0],
                     dtype=torch.float32, device=device)
    dist.broadcast(t, src=mesh.rank_of(pp=mesh.pp - 1), group=mesh.pp_group)
    return t.item()


# ---------------------------------------------------------------------
# 3. one micro-batch, forward and backward
# ---------------------------------------------------------------------
#: Seconds this rank spent *computing* rather than waiting for a
#: neighbor. 
COMPUTE_SECONDS = 0.0


def _sync(device):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_compute_timer() -> None:
    global COMPUTE_SECONDS
    COMPUTE_SECONDS = 0.0


def forward_step(stage, batch, inp):
    """Run one micro-batch through this stage.

    `inp` is the activation received from the previous stage, or None on
    the first stage, where the batch's token ids are the input instead.
    Returns `(inp, out, loss)`, which is everything the matching
    backward will need.
    """
    x, y = batch
    src = x if inp is None else inp
    global COMPUTE_SECONDS
    _sync(src.device)
    t0 = time.perf_counter()
    # Autocast to use bf16 on a GPU, fp32 on a CPU
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=src.is_cuda):
        out, loss = stage(src, y)
    _sync(src.device)
    COMPUTE_SECONDS += time.perf_counter() - t0
    return (inp, out, loss)


def backward_step(saved, grad, scale):
    """Run the backward of one micro-batch, and return its input gradient.

    On the last stage there is no incoming gradient and `loss` is real,
    so the backward starts from the loss. Everywhere else the gradient
    arrives from the stage in front.
    """
    global COMPUTE_SECONDS
    inp, out, loss = saved
    device = out.device if out is not None else None
    _sync(device)
    t0 = time.perf_counter()
    if loss is not None:
        (loss / scale).backward()
    else:
        # The wire dtype need not match the activation's: under autocast
        # the hidden states are bf16 while a stage's own may be fp32.
        torch.autograd.backward(out, grad.to(out.dtype))
    _sync(device)
    COMPUTE_SECONDS += time.perf_counter() - t0
    return None if inp is None else inp.grad


# ---------------------------------------------------------------------
# 4. all-forward-all-backward 
# ---------------------------------------------------------------------
def all_forward_all_backward(stage, pipe, micro_batches, scale):
    """Every micro-batch forward, then every micro-batch backward.

    Read this before writing 1F1B. 
    """
    saved, losses = [], []

    for batch in micro_batches:
        inp = pipe.recv_forward()
        rec = forward_step(stage, batch, inp)
        pipe.send_forward(rec[1])
        saved.append(rec)
        if rec[2] is not None:
            losses.append(rec[2].item())

    for rec in reversed(saved):
        grad = pipe.recv_backward()
        pipe.send_backward(backward_step(rec, grad, scale))

    return (sum(losses) / len(losses)) if losses else None


# ---------------------------------------------------------------------
# 5. one-forward-one-backward 
# ---------------------------------------------------------------------
def one_forward_one_backward(stage, pipe, micro_batches, scale):
    """Interleave, so activation memory stops growing with `m` 
    (the number of micro-batches).

    Same operations as `all_forward_all_backward`.
    Reduce the number of activation graphs alive at once
    to at most `p` (the number of pipeline stages). There are three phases.

    * **Warm up.** Stage `i` runs `p - 1 - i` forwards before its first
      backward to fill the pipe.
      `len(micro_batches)` may be smaller than the 
      warm-up length. In this case clamp it to the number of micro-batches.
    * **Steady state.** For each remaining micro-batch: one forward, then
      one backward of the *oldest* forward still outstanding.
    * **Cool down.** No forwards left; drain the outstanding backwards.

    Four things to note:

    1. **Backward in forward order.** 
        Be sure to apply the backward pass in the same order as the forwards.
    2. **Use the combined transfers in the steady state.**
       `send_forward_recv_backward` and `send_backward_recv_forward`
       exist because doing those as separate blocking causes deadlocks.
       See `Pipe`.
    3. **`forward_step` and `backward_step` do all the tensor work.** You
       should not need to write any autograd code.
    4. **Return what AFAB returns**. Return the mean loss on the last stage,
       None elsewhere.

    """
    p, i = pipe.mesh.pp, pipe.mesh.pp_rank
    m = len(micro_batches)

    raise NotImplementedError("Part 3: one_forward_one_backward")


SCHEDULES = {
    "afab": all_forward_all_backward,
    "1f1b": one_forward_one_backward,
}


# ---------------------------------------------------------------------
# 6. putting it together
# ---------------------------------------------------------------------
def pipeline_step(stage, mesh, micro_batches, schedule="1f1b", device="cpu",
                  dtype=torch.float32, scale=None):
    """Run one optimizer step's worth of micro-batches through the pipe.
    """
    if not micro_batches:
        return None
    micro = micro_batches[0][0].shape[0]
    pipe = Pipe(mesh, stage.activation_shape(micro), dtype, device)
    reset_compute_timer()
    loss = SCHEDULES[schedule](stage, pipe, micro_batches,
                              scale if scale is not None else len(micro_batches))
    return broadcast_loss(loss, mesh, device), COMPUTE_SECONDS
