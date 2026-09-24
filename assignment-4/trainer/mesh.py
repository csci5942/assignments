"""Ranks arranged into a (dp, tp, pp) grid, and the groups that go with it.

Given. You should not need to change anything here, but you do need to
read it, because one thing in this file is a decision rather than a fact:
the **order of the axes**.

A mesh of `dp x tp x pp` ranks has to be laid out on a line of global
ranks somewhere, and the layout decides which dimension's traffic
crosses which link. Slurm numbers ranks so that adjacent ranks land on
the same node, so the *last* axis in `order` is the one whose group
members are neighbours -- it gets the fast link. The first axis is the
one that gets split across nodes.

    order = ("pp", "dp", "tp")     # the default
      rank = pp_rank * (dp * tp) + dp_rank * tp + tp_rank

With eight ranks on two nodes as (dp=2, tp=2, pp=2), that default puts
each tensor parallel pair on the same node and splits the pipeline
across the node boundary. Whether that is the right call is Part 4's
question, and `--axis-order` is how you answer it.

    python -m trainer.mesh --dp 2 --tp 2 --pp 2 --axis-order pp,dp,tp

prints the grid and says which dimensions straddle the boundary, without
launching anything.
"""

from __future__ import annotations

import itertools
import os
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist

#: The three axes, in a fixed order used only for deterministic iteration.
#: Every rank must create every process group in the same sequence, or the
#: collectives deadlock; iterating this constant rather than the
#: user-supplied order guarantees it.
AXES = ("pp", "dp", "tp")

#: Slowest-varying axis first. Tensor parallel is last, so its ranks are
#: adjacent and stay inside a node -- it is the dimension that talks most
#: often and can hide it least.
DEFAULT_ORDER = ("pp", "dp", "tp")


# ---------------------------------------------------------------------
# the environment
# ---------------------------------------------------------------------
def _env_int(*names: str, default: int = 0) -> int:
    for n in names:
        if os.environ.get(n):
            return int(os.environ[n])
    return default


def env_rank() -> int:
    """This process's global rank. `srun` sets SLURM_PROCID, torchrun RANK."""
    return _env_int("RANK", "SLURM_PROCID", default=0)


def env_world_size() -> int:
    return _env_int("WORLD_SIZE", "SLURM_NTASKS", default=1)


def env_local_rank() -> int:
    """Index of this process's GPU on its own node."""
    return _env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)


def _default_master_addr() -> str:
    """The first host in the allocation, which every rank can agree on."""
    nodelist = (os.environ.get("SLURM_JOB_NODELIST")
                or os.environ.get("SLURM_NODELIST"))
    if nodelist:
        try:
            out = subprocess.run(["scontrol", "show", "hostnames", nodelist],
                                 capture_output=True, text=True, timeout=10)
            hosts = out.stdout.split()
            if hosts:
                return hosts[0]
        except (OSError, subprocess.SubprocessError):
            pass
    return "127.0.0.1"


#: The timeout `init_distributed` was asked for, remembered so that the
#: groups `Mesh` builds later get it too. `new_group` does *not* inherit
#: the default group's timeout -- it falls back to NCCL's own 10 minutes
#: -- so without this every collective in Parts 2 to 4 runs on a shorter
#: fuse than the one the job asked for, and a slow multi-node startup is
#: reported as a hang.
_GROUP_TIMEOUT: timedelta = timedelta(minutes=30)


def init_distributed(backend: str | None = None,
                     timeout_minutes: int = 30) -> bool:
    """Bring up the default process group from the environment.

    Idempotent, and a no-op at world size 1, so the single-GPU path in
    `train.py` keeps working with no distributed machinery at all.
    Returns True if this process ended up in a real process group.
    """
    global _GROUP_TIMEOUT
    _GROUP_TIMEOUT = timedelta(minutes=timeout_minutes)
    if dist.is_available() and dist.is_initialized():
        return True
    world = env_world_size()
    if world <= 1:
        return False
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    os.environ.setdefault("MASTER_ADDR", _default_master_addr())
    os.environ.setdefault("MASTER_PORT", "29500")
    # torchrun sets these two; `srun` does not, and init_process_group
    # reads them rather than the SLURM_* names.
    os.environ["RANK"] = str(env_rank())
    os.environ["WORLD_SIZE"] = str(world)
    kwargs = {}
    if backend == "nccl":
        # Bind this rank to its GPU *before* the process group exists, and
        # tell the group which device it owns. Without `device_id`, NCCL
        # guesses from the global rank -- which is right on a uniform
        # allocation and hangs on a heterogeneous one, and makes every
        # barrier print a warning in the meantime.
        local = env_local_rank()
        torch.cuda.set_device(local)
        kwargs["device_id"] = torch.device(f"cuda:{local}")
    dist.init_process_group(backend=backend,
                            timeout=timedelta(minutes=timeout_minutes),
                            **kwargs)
    return True


# ---------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------
def _rank_of(coords: dict[str, int], sizes: dict[str, int],
             order: tuple[str, ...]) -> int:
    """Global rank at the given coordinates. The first axis varies slowest."""
    r = 0
    for axis in order:
        r = r * sizes[axis] + coords[axis]
    return r


def _coords_of(rank: int, sizes: dict[str, int],
               order: tuple[str, ...]) -> dict[str, int]:
    """Inverse of `_rank_of`."""
    coords = {}
    for axis in reversed(order):
        coords[axis] = rank % sizes[axis]
        rank //= sizes[axis]
    return coords


def parse_order(text: str) -> tuple[str, ...]:
    """`"pp,dp,tp"` -> `("pp", "dp", "tp")`, with a readable error."""
    parts = tuple(p.strip() for p in text.split(",") if p.strip())
    if sorted(parts) != sorted(AXES):
        raise ValueError(
            f"axis order {text!r} must be a permutation of "
            f"{','.join(AXES)}, got {list(parts)}")
    return parts


class Mesh:
    """The rank grid, this rank's place in it, and the groups it belongs to.

    Use `Mesh.from_env(...)` in real code. The plain constructor takes
    `rank` and `world_size` directly and can build the arithmetic with no
    process group at all, which is how `tests/test_mesh.py` checks the
    layout on a laptop and how `python -m trainer.mesh` prints it.
    """

    def __init__(self, dp: int = 1, tp: int = 1, pp: int = 1,
                 order: tuple[str, ...] | str = DEFAULT_ORDER, *,
                 rank: int | None = None, world_size: int | None = None,
                 local_rank: int | None = None, create_groups: bool = True):
        if isinstance(order, str):
            order = parse_order(order)
        order = tuple(order)
        if sorted(order) != sorted(AXES):
            raise ValueError(f"axis order must be a permutation of {AXES}, "
                             f"got {order}")
        for name, n in (("dp", dp), ("tp", tp), ("pp", pp)):
            if n < 1:
                raise ValueError(f"{name}={n} must be at least 1")

        self.sizes = {"dp": dp, "tp": tp, "pp": pp}
        self.order = order
        self.rank = env_rank() if rank is None else rank
        self.world_size = env_world_size() if world_size is None else world_size
        self.local_rank = env_local_rank() if local_rank is None else local_rank

        if dp * tp * pp != self.world_size:
            raise ValueError(
                f"dp*tp*pp = {dp}*{tp}*{pp} = {dp * tp * pp} does not match "
                f"world size {self.world_size}. Every rank needs a place in "
                f"the grid, and no place may hold two ranks.")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank {self.rank} is outside world size {self.world_size}")

        self.coords = _coords_of(self.rank, self.sizes, order)
        self.groups: dict[str, object | None] = {a: None for a in AXES}
        if create_groups and dist.is_available() and dist.is_initialized():
            self._build_groups()

    # -- groups -------------------------------------------------------
    def _build_groups(self) -> None:
        """Create every group on every rank, in the same order everywhere.

        `new_group` is collective: all ranks must call it for all groups,
        including the ones they are not members of. Groups of size one
        are built too, so `mesh.dp_group` is always a real group and
        callers can pass `group=mesh.dp_group` unconditionally. The
        alternative -- leaving it None and letting the collective fall
        back to WORLD -- is correct only while tp and pp are both 1, and
        silently wrong in Part 4.

        Each group is given `_GROUP_TIMEOUT` explicitly, because
        `new_group` does not inherit one.
        """
        for axis in AXES:                    # fixed order, not self.order
            others = [a for a in AXES if a != axis]
            spans = itertools.product(*(range(self.sizes[a]) for a in others))
            for combo in spans:
                coords = dict(zip(others, combo))
                ranks = [_rank_of({**coords, axis: i}, self.sizes, self.order)
                         for i in range(self.sizes[axis])]
                group = dist.new_group(ranks, timeout=_GROUP_TIMEOUT)
                if self.rank in ranks:
                    self.groups[axis] = group

    # -- accessors ----------------------------------------------------
    @property
    def dp(self) -> int:
        return self.sizes["dp"]

    @property
    def tp(self) -> int:
        return self.sizes["tp"]

    @property
    def pp(self) -> int:
        return self.sizes["pp"]

    @property
    def dp_rank(self) -> int:
        return self.coords["dp"]

    @property
    def tp_rank(self) -> int:
        return self.coords["tp"]

    @property
    def pp_rank(self) -> int:
        return self.coords["pp"]

    @property
    def dp_group(self):
        return self.groups["dp"]

    @property
    def tp_group(self):
        return self.groups["tp"]

    @property
    def pp_group(self):
        return self.groups["pp"]

    @property
    def device(self) -> str:
        return f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"

    @property
    def is_leader(self) -> bool:
        """Global rank 0: the one that prints and writes the log."""
        return self.rank == 0

    def rank_of(self, **coords: int) -> int:
        """Global rank at these coordinates; unnamed axes keep ours."""
        unknown = set(coords) - set(AXES)
        if unknown:
            raise ValueError(f"unknown axes {sorted(unknown)}")
        return _rank_of({**self.coords, **coords}, self.sizes, self.order)

    def group_ranks(self, axis: str) -> list[int]:
        """The global ranks sharing this rank's group along `axis`."""
        if axis not in AXES:
            raise ValueError(f"unknown axis {axis!r}")
        return [self.rank_of(**{axis: i}) for i in range(self.sizes[axis])]

    # -- pipeline neighbours ------------------------------------------
    @property
    def is_first_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.pp_rank == self.pp - 1

    @property
    def prev_stage(self) -> int | None:
        return None if self.is_first_stage else self.rank_of(pp=self.pp_rank - 1)

    @property
    def next_stage(self) -> int | None:
        return None if self.is_last_stage else self.rank_of(pp=self.pp_rank + 1)

    # -- lifecycle -----------------------------------------------------
    @classmethod
    def from_env(cls, dp: int = 1, tp: int = 1, pp: int = 1,
                 order: tuple[str, ...] | str = DEFAULT_ORDER,
                 backend: str | None = None) -> "Mesh":
        """Initialize the process group if needed, then build the mesh."""
        init_distributed(backend)
        return cls(dp=dp, tp=tp, pp=pp, order=order)

    @staticmethod
    def shutdown() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def barrier(self) -> None:
        """Wait for every rank, and flush this one's GPU work first.

        The flush is the part that matters, and it is not obvious.
        NCCL collectives are asynchronous: `broadcast_module_state`
        returns once its several hundred broadcasts are *enqueued*, not
        once they have run. A barrier issued behind them goes on a
        different communicator, and ranks do not reach it together --
        on two nodes we measured the far node taking 2.2 seconds to
        enqueue what the near node enqueued in 0.12. The ranks that get
        there first then wait inside NCCL for ranks whose GPUs are
        still working through the earlier collective, and the whole
        thing deadlocks: no error, no progress, and nothing in the log
        until the watchdog fires ten minutes later.

        Synchronising first makes the barrier mean what it says --
        everyone has finished, not everyone has asked -- and costs
        nothing, because a barrier is already a synchronisation point.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            torch.cuda.synchronize()
            dist.barrier(device_ids=[self.local_rank])
        else:
            dist.barrier()

    def __repr__(self) -> str:
        grid = " x ".join(f"{a}={self.sizes[a]}" for a in self.order)
        here = " ".join(f"{a}={self.coords[a]}" for a in self.order)
        return (f"Mesh({grid}, order={'/'.join(self.order)}) "
                f"rank {self.rank}/{self.world_size} at [{here}]")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="show a mesh layout without launching anything")
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--axis-order", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--gpus-per-node", type=int, default=4)
    args = ap.parse_args()

    world = args.dp * args.tp * args.pp
    order = parse_order(args.axis_order)
    meshes = [Mesh(args.dp, args.tp, args.pp, order, rank=r, world_size=world,
                   create_groups=False) for r in range(world)]

    print(f"world {world}, {args.gpus_per_node} GPUs per node, "
          f"axis order {'/'.join(order)}\n")
    for r, m in enumerate(meshes):
        print(f"  rank {r}  node {r // args.gpus_per_node}  "
              + "  ".join(f"{a}={m.coords[a]}" for a in order))

    print()
    for axis in AXES:
        if meshes[0].sizes[axis] == 1:
            print(f"  {axis}: size 1, no communication")
            continue
        straddling = sum(
            len({x // args.gpus_per_node for x in m.group_ranks(axis)}) > 1
            for m in meshes)
        verdict = ("crosses the node boundary" if straddling
                   else "stays inside one node")
        print(f"  {axis}: {verdict}")
