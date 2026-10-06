"""Pass/fail lines for everything Assignment 5 needs, and what to do about
a failure. Run it once after `pip install -r requirements.txt`."""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True


def check(label, good, fix=""):
    global ok
    print(f"[{'ok' if good else 'FAIL'}] {label}" + ("" if good else f"\n       {fix}"))
    ok = ok and good


check(f"python {sys.version.split()[0]} >= 3.10", sys.version_info >= (3, 10),
      "use python3.10 or newer")
for mod in ("torch", "numpy", "tiktoken", "safetensors", "huggingface_hub", "sacrebleu", "matplotlib"):
    try:
        importlib.import_module(mod)
        check(f"import {mod}", True)
    except ImportError:
        check(f"import {mod}", False, "pip install -r requirements.txt")
try:
    import torch
    cuda = torch.cuda.is_available()
    check("CUDA GPU", cuda, "Parts 1, 3, 4 and 5 need a GPU: Delta's A40s (A4), the course cluster, or a cloud VM")
    if cuda:
        free, total = torch.cuda.mem_get_info()
        name = torch.cuda.get_device_name()
        check(f"{name}, {total / 2**30:.0f} GiB ({free / 2**30:.0f} free)", total >= 12 * 2**30,
              "gpt2-medium full fine-tuning peaks near 10 GiB at the default batch; use a smaller batch or gpt2")
except ImportError:
    pass
check("data/e2e/train.jsonl", os.path.exists(os.path.join(HERE, "data", "e2e", "train.jsonl")),
      "python data/prepare_e2e.py")
check("data/fineweb_val.bin", os.path.exists(os.path.join(HERE, "data", "fineweb_val.bin")),
      "it ships with the repo; re-clone if it is missing")
sys.exit(0 if ok else 1)
