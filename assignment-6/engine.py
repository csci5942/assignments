"""Continuous batching (Part 3).

A small serving engine over your paged KV cache. Every iteration the
scheduler retires finished requests and admits waiting ones. The engine
then prefills the newly admitted requests, one at a time, and runs one
batched decode step for everything running. 

Two things for you to implement in this file.

    Scheduler.can_admit    does the pool have room for this request?
    Scheduler.schedule     retire finished requests, admit waiting ones

`Scheduler.preempt`, which evicts a request when blocks run out, is
given.

To run the correctness suite

    python -m pytest tests/test_engine.py -q
"""

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from model import GPT
from paged import PagedKVCache


@dataclass
class Request:
    """One request: generate exactly `max_new` tokens after `prompt`.

    `out` collects the generated tokens, and the request is `done` once it
    has `max_new` of them (EOS is ignored). `admitted` and `finished` record
    the iteration of each, and `preemptions` counts how often it was
    evicted.
    """
    id: int
    prompt: list[int]
    max_new: int                  # generate exactly this many tokens (EOS ignored)
    out: list[int] = field(default_factory=list)
    admitted: int = -1            # iteration of (last) admission
    finished: int = -1            # iteration it finished
    t_done: float = 0.0           # seconds from the start of the run
    preemptions: int = 0

    @property
    def done(self) -> bool:
        """Whether all `max_new` tokens have been generated."""
        return len(self.out) >= self.max_new


def blocks_for(n_tokens: int, block_size: int) -> int:
    """Blocks needed to hold `n_tokens`, rounding up."""
    return -(-n_tokens // block_size)


class Scheduler:
    """Continuous batching, first come, first served.

    * `self.waiting`: requests not yet admitted, in arrival order.
    * `self.running`: admitted requests, in admission order. Once the engine
      prefills one, it has a sequence in the cache under its id.
    * `self.finished`: retired requests, in the order they finished.

    At most `max_batch` requests run at once, and the pool must not run out
    of blocks in the middle of a decode step. How `can_admit` makes sure of
    that depends on `optimistic`:

    **Safe admission** (`optimistic=False`, the default). Admit a request
    only if the pool can hold every running request at its full length,
    `len(prompt) + max_new`, and this one at its full length too. Each
    request's `max_new` is known in advance, so a decode step never runs
    out.

    **Optimistic admission** (`optimistic=True`). Count only what each
    request needs for its next step, `len(prompt) + len(out) + 1`. More
    requests run at once, and when a decode step does run out, the engine
    calls `preempt` (given) to free some blocks.
    """

    def __init__(self, cache: PagedKVCache, max_batch: int, optimistic: bool = False):
        self.cache = cache
        self.max_batch = max_batch
        self.optimistic = optimistic
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []

    def add(self, req: Request):
        self.waiting.append(req)

    def can_admit(self, req: Request) -> bool:
        """Whether the pool has room for `req`, next to the running requests.

        The tokens `r` must be able to hold depends on the scheduler: 
        its full length, `len(r.prompt) + r.max_new`, or with `optimistic`,
        `len(r.prompt) + len(r.out) + 1`. 

        What to write: return whether the blocks `req` needs fit in what is
        left in the pool after accounting for the blocks owed to running
        requests. 
        """
        bs = self.cache.block_size
        free = self.cache.allocator.num_free
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    def schedule(self, it: int) -> list[Request]:
        """Retire finished requests, then admit waiting ones.

        Called at the start of iteration `it`. Returns the requests admitted
        this iteration, in order.

        What to write:

        * Retire every running request that is `done`. Update necessary state
          in the Scheduler and in the cache.

        * Admit from the front of `self.waiting` while there is a batch slot
          and the request can be admitted. Schedule in FCFS order.
        """
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    def preempt(self) -> Request:
        """Evict a running request to free its blocks (given).
        """
        r = next(r for r in reversed(self.running) if not r.done)
        self.running.remove(r)
        self.cache.remove(r.id)
        r.preemptions += 1
        self.waiting.appendleft(r)
        return r


class StaticScheduler(Scheduler):
    """Static batching (given), for comparison.

    Admits up to `max_batch` requests that fit at full length, only when
    nothing is running, and runs them all to completion.
    """

    def schedule(self, it: int) -> list[Request]:
        for r in [r for r in self.running if r.done]:
            self.cache.remove(r.id)
            self.running.remove(r)
            r.finished = it
            self.finished.append(r)
        if self.running:
            return []
        self.optimistic = False
        admitted = []
        while self.waiting and len(admitted) < self.max_batch and self.can_admit(self.waiting[0]):
            r = self.waiting.popleft()
            r.admitted = it
            self.running.append(r)
            admitted.append(r)
        return admitted


class Engine:
    """The serving loop (given): `run` calls your scheduler every iteration."""
    def __init__(self, model: GPT, cache: PagedKVCache, scheduler: Scheduler):
        self.model, self.cache, self.sched = model, cache, scheduler
        self.dev = cache.k[0].device
        self.batch_sizes: list[int] = []   # decode batch size per iteration

    @torch.no_grad()
    def run(self, requests: list[Request]) -> list[Request]:
        """Greedy-decode every request to max_new tokens. Returns them in
        the order they finished."""
        for r in requests:
            self.sched.add(r)
        t0, it = time.perf_counter(), 0
        while self.sched.waiting or self.sched.running:
            for r in self.sched.schedule(it):
                self.prefill(r)
            self.decode()
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter() - t0
            for r in self.sched.running:
                if r.done and not r.t_done:
                    r.t_done = now
            it += 1
            if not self.sched.running and self.sched.waiting:
                raise RuntimeError(f"request {self.sched.waiting[0].id} can never fit in the pool")
        return self.sched.finished

    def prefill(self, r: Request):
        """Run the prompt (plus any tokens from before a preemption) and
        take the next token from the last position."""
        self.cache.add(r.id)
        self.cache.select([r.id])
        ids = torch.tensor([r.prompt + r.out], device=self.dev)
        logits, _ = self.model(ids, cache=self.cache)
        r.out.append(int(logits[0, -1].argmax()))

    def decode(self):
        """One token for every running request that still needs one. If the
        pool runs out, preempt and retry with a smaller batch."""
        while True:
            batch = [r for r in self.sched.running if not r.done and len(r.out) > 0]
            if not batch:
                return
            self.cache.select([r.id for r in batch])
            ids = torch.tensor([[r.out[-1]] for r in batch], device=self.dev)
            try:
                logits, _ = self.model(ids, cache=self.cache)
                break
            except MemoryError:
                if not self.sched.optimistic:
                    raise
                self.sched.preempt()
        self.batch_sizes.append(len(batch))
        for r, nxt in zip(batch, logits[:, -1].argmax(-1).tolist()):
            r.out.append(nxt)
