"""Part 2: the group-relative advantage (CPU, no model)."""
import math

from grpo import group_advantage


def test_binary_rewards_give_plus_and_minus_one():
    adv = group_advantage([1, 0, 0, 1])          # mean 0.5, std 0.5
    assert [round(a, 3) for a in adv] == [1.0, -1.0, -1.0, 1.0]


def test_a_flat_group_gives_zero():
    assert group_advantage([7, 7, 7, 7]) == [0.0, 0.0, 0.0, 0.0]


def test_scale_free():
    a = group_advantage([3, 5, 9, 7])
    b = group_advantage([30, 50, 90, 70])
    assert all(math.isclose(x, y, abs_tol=1e-3) for x, y in zip(a, b))
    assert math.isclose(sum(a), 0.0, abs_tol=1e-6)
