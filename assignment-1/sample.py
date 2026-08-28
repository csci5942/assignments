"""Generate text from a trained checkpoint.

    python sample.py --run out/base --prompt "DUKE OF" --tokens 400
"""

import argparse
import json
import os

import torch

from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing ckpt.pt")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join(args.run, "ckpt.pt"), map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)

    with open(os.path.join(HERE, "data", "meta.json")) as f:
        meta = json.load(f)
    itos = meta["itos"]
    stoi = {ch: i for i, ch in enumerate(itos)}

    idx = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print("".join(itos[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
