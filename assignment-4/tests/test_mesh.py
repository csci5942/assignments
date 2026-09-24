"""The device mesh. Given code, so these pass before you write anything.

If one of these fails, something is wrong with the handout rather than
with you -- open an issue.

Most of the file needs no processes at all: the rank/coordinate mapping
is arithmetic, and `Mesh(..., create_groups=False)` exercises it for a
world that does not exist. The last two tests fork four gloo ranks and
check that the process groups really are what the arithmetic promised.
"""

import pytest

torch = pytest.importorskip("torch")

from trainer.mesh import (AXES, DEFAULT_ORDER, Mesh, _coords_of,  # noqa: E402
                          _rank_of, parse_order)

from conftest import run_distributed  # noqa: E402


def mesh_at(rank, dp=1, tp=1, pp=1, order=DEFAULT_ORDER):
    return Mesh(dp=dp, tp=tp, pp=pp, order=order, rank=rank,
                world_size=dp * tp * pp, create_groups=False)


# ---------------------------------------------------------------------
# the layout
# ---------------------------------------------------------------------
def test_trivial_mesh_is_the_single_gpu_case():
    m = mesh_at(0)
    assert (m.dp, m.tp, m.pp) == (1, 1, 1)
    assert (m.dp_rank, m.tp_rank, m.pp_rank) == (0, 0, 0)
    assert m.is_leader and m.is_first_stage and m.is_last_stage
    assert m.prev_stage is None and m.next_stage is None


@pytest.mark.parametrize("order", [("pp", "dp", "tp"), ("dp", "tp", "pp"),
                                   ("tp", "pp", "dp")])
def test_every_rank_has_exactly_one_place(order):
    """The grid is a bijection: 8 ranks, 8 distinct coordinate triples."""
    sizes = {"dp": 2, "tp": 2, "pp": 2}
    seen = {}
    for r in range(8):
        coords = _coords_of(r, sizes, order)
        key = tuple(coords[a] for a in AXES)
        assert key not in seen, \
            f"ranks {seen[key]} and {r} both land on {key} under {order}"
        seen[key] = r
        assert _rank_of(coords, sizes, order) == r, \
            "_rank_of and _coords_of disagree"
    assert len(seen) == 8


def test_last_axis_is_the_contiguous_one():
    """The point of the whole file: the last axis gets adjacent ranks.

    Slurm packs adjacent ranks onto the same node, so whichever axis is
    last in `order` is the one that stays on the fast link. This is the
    knob Part 4's placement experiment turns.
    """
    tp_last = [mesh_at(r, dp=2, tp=2, pp=2, order=("pp", "dp", "tp"))
               for r in range(8)]
    assert tp_last[0].group_ranks("tp") == [0, 1]
    assert tp_last[0].group_ranks("dp") == [0, 2]
    assert tp_last[0].group_ranks("pp") == [0, 4]

    tp_first = [mesh_at(r, dp=2, tp=2, pp=2, order=("tp", "dp", "pp"))
                for r in range(8)]
    assert tp_first[0].group_ranks("pp") == [0, 1]
    assert tp_first[0].group_ranks("tp") == [0, 4]

    # With four GPUs per node, the axis whose partners are four apart is
    # the one paying for the network.
    def straddles(meshes, axis, per_node=4):
        return any(len({r // per_node for r in m.group_ranks(axis)}) > 1
                   for m in meshes)

    assert straddles(tp_last, "pp") and not straddles(tp_last, "tp")
    assert straddles(tp_first, "tp") and not straddles(tp_first, "pp")


def test_group_membership_is_symmetric():
    """If a is in b's group, b is in a's -- otherwise the group deadlocks."""
    meshes = [mesh_at(r, dp=2, tp=2, pp=2) for r in range(8)]
    for axis in AXES:
        for m in meshes:
            for peer in m.group_ranks(axis):
                assert m.rank in meshes[peer].group_ranks(axis), \
                    f"rank {m.rank} and {peer} disagree about their {axis} group"


def test_pipeline_neighbours_chain():
    meshes = [mesh_at(r, pp=4) for r in range(4)]
    assert [m.pp_rank for m in meshes] == [0, 1, 2, 3]
    assert meshes[0].prev_stage is None and meshes[3].next_stage is None
    for r in range(3):
        assert meshes[r].next_stage == r + 1
        assert meshes[r + 1].prev_stage == r


def test_rank_of_keeps_unnamed_axes():
    m = mesh_at(3, dp=2, tp=2, pp=2)          # pp=0, dp=1, tp=1
    assert (m.pp_rank, m.dp_rank, m.tp_rank) == (0, 1, 1)
    assert m.rank_of() == 3
    assert m.rank_of(tp=0) == 2
    assert m.rank_of(pp=1) == 7


# ---------------------------------------------------------------------
# the errors
# ---------------------------------------------------------------------
def test_mismatched_world_is_refused():
    with pytest.raises(ValueError, match="world size"):
        Mesh(dp=2, tp=2, pp=2, rank=0, world_size=4, create_groups=False)


def test_bad_axis_order_is_refused():
    with pytest.raises(ValueError, match="permutation"):
        parse_order("dp,tp")
    with pytest.raises(ValueError, match="permutation"):
        parse_order("dp,tp,dp")


def test_parse_order_round_trips():
    assert parse_order(" pp , dp , tp ") == ("pp", "dp", "tp")
    assert parse_order(",".join(DEFAULT_ORDER)) == DEFAULT_ORDER


def test_rank_of_rejects_unknown_axis():
    with pytest.raises(ValueError, match="unknown axes"):
        mesh_at(0).rank_of(cp=1)


# ---------------------------------------------------------------------
# the real process groups
# ---------------------------------------------------------------------
def _check_groups(rank, world):
    """dp=2, tp=2: an all-reduce on one axis must not touch the other."""
    import torch.distributed as dist
    from trainer.mesh import Mesh

    mesh = Mesh(dp=2, tp=2, pp=1, rank=rank, world_size=world)
    assert dist.get_world_size(mesh.dp_group) == 2
    assert dist.get_world_size(mesh.tp_group) == 2
    assert dist.get_world_size(mesh.pp_group) == 1, \
        "a size-1 axis must still be a real group, not None"

    # Each rank contributes 1 << rank, so the sum names its members exactly.
    t = torch.tensor([float(1 << rank)])
    dist.all_reduce(t, group=mesh.dp_group)
    expected = float(sum(1 << r for r in mesh.group_ranks("dp")))
    assert t.item() == expected, (
        f"rank {rank}: all-reduce over the dp group summed "
        f"{int(t.item()):#b}, expected {int(expected):#b} for members "
        f"{mesh.group_ranks('dp')}")


def test_process_groups_match_the_arithmetic():
    run_distributed(_check_groups, 4)


def _check_order_changes_partners(rank, world):
    """Same (dp, tp), different order, different dp partners."""
    import torch.distributed as dist
    from trainer.mesh import Mesh

    a = Mesh(dp=2, tp=2, pp=1, order=("pp", "dp", "tp"),
             rank=rank, world_size=world)
    assert a.group_ranks("dp") == [rank % 2, rank % 2 + 2]

    b = Mesh(dp=2, tp=2, pp=1, order=("pp", "tp", "dp"),
             rank=rank, world_size=world)
    assert b.group_ranks("dp") == [rank - rank % 2, rank - rank % 2 + 1]

    t = torch.tensor([float(1 << rank)])
    dist.all_reduce(t, group=b.dp_group)
    assert t.item() == float(sum(1 << r for r in b.group_ranks("dp")))


def test_axis_order_changes_who_talks_to_whom():
    run_distributed(_check_order_changes_partners, 4)
