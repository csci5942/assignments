"""Pass/fail lines for everything Assignment 6 needs, and what to do about
a failure. Run it once after `pip install -r requirements.txt`."""

import importlib
import shutil
import sys

ok = True


def check(label, good, fix=""):
    global ok
    print(f"[{'ok' if good else 'FAIL'}] {label}" + ("" if good else f"\n       {fix}"))
    ok = ok and good


check(f"python {sys.version.split()[0]} >= 3.10", sys.version_info >= (3, 10),
      "use python3.10 or newer")
for mod in ("torch", "numpy", "tiktoken", "safetensors", "huggingface_hub", "pytest"):
    try:
        importlib.import_module(mod)
        check(f"import {mod}", True)
    except ImportError:
        check(f"import {mod}", False, "pip install -r requirements.txt")
try:
    import torch
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    check(f"torch {torch.__version__} >= 2.5", (major, minor) >= (2, 5),
          "scaled_dot_product_attention must return 0, not NaN, for a fully masked row: "
          "pip install -U torch")
    cuda = torch.cuda.is_available()
    check("CUDA GPU", cuda, "the tests run on a CPU, but sample.py and the benchmarks need a GPU")
    if cuda:
        free, total = torch.cuda.mem_get_info()
        name = torch.cuda.get_device_name()
        check(f"{name}, {total / 2**30:.0f} GiB ({free / 2**30:.0f} free)", total >= 15 * 2**30,
              "the benchmarks peak near 11 GB (gpt2-xl in bf16 plus up to 4 GB of KV "
              "cache); use a GPU with at least 16 GB")
        check("bf16", torch.cuda.is_bf16_supported(),
              "the benchmarks need bf16: use an Ampere or newer GPU (A40, A100, L4, ...)")
except ImportError:
    pass
free_disk = shutil.disk_usage(".").free
check(f"{free_disk / 1e9:.0f} GB free disk", free_disk > 10e9,
      "gpt2-xl's weights are 6.4 GB in ~/.cache/huggingface; resize the VM's disk")
sys.exit(0 if ok else 1)
