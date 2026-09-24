"""Environment tests. These must pass before you start Part 1.

    python -m pytest tests/test_env.py -q

`python check_env.py` covers the same ground with friendlier output and
some advisory checks; this file is the version that gates the graded
test run.
"""

import sys

import pytest


def test_python_version():
    assert sys.version_info >= (3, 10), (
        f"Python 3.10+ required, found {sys.version.split()[0]}. "
        "On Delta: module load pytorch-conda/2.12 && conda activate base")


def test_torch_importable():
    torch = pytest.importorskip("torch")
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    assert (major, minor) >= (2, 2), (
        f"torch 2.2+ required for the distributed APIs this assignment "
        f"uses, found {torch.__version__}")


def test_distributed_available():
    import torch.distributed as dist
    assert dist.is_available(), "this PyTorch was built without torch.distributed"
    assert dist.is_gloo_available(), (
        "gloo is unavailable, so the CPU correctness tests in Parts 1 to 3 "
        "cannot run")
