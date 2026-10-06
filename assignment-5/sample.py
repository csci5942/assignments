"""Write a few tokens from a GPT-2 model, pretrained or fine-tuned.

    python sample.py                                   # gpt2, a plain prompt
    python sample.py --model gpt2-medium --prompt "The capital of France is"
    python sample.py --run out/lora-r4 --mr "name[The Vaults], eatType[pub]"

Greedy decoding, as in eval.py. Given.
"""

import argparse
import json
import os

import torch

import lora
import sft
from model import from_pretrained

MAX_NEW = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--run", default=None, help="an out/<name> directory; overrides --model")
    ap.add_argument("--prompt", default="My favourite restaurant in the city centre is")
    ap.add_argument("--mr", default=None, help="an E2E meaning representation, formatted with the template")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.run:
        with open(os.path.join(args.run, "summary.json")) as f:
            cfg = json.load(f)["config"]
        model = from_pretrained(cfg["model"])
        if cfg["mode"] == "lora":
            lora.apply_lora(model, cfg["rank"], cfg["alpha"], tuple(cfg["targets"]))
            model.load_state_dict(torch.load(os.path.join(args.run, "lora.pt"))["lora"], strict=False)
            lora.merge_lora(model)
        else:
            model.load_state_dict(torch.load(os.path.join(args.run, "ckpt.pt"))["model"])
    else:
        model = from_pretrained(args.model)
    model = model.to(args.device).eval()

    if args.mr:
        ids, _ = sft.format_example(args.mr, "")
    else:
        ids = sft.ENC.encode(args.prompt)
    x = torch.tensor([ids], device=args.device)
    with torch.no_grad():
        for _ in range(MAX_NEW):
            logits, _ = model(x)
            nxt = logits[0, -1].argmax().item()
            if nxt == sft.EOT:
                break
            x = torch.cat([x, torch.tensor([[nxt]], device=args.device)], dim=1)
    print(sft.ENC.decode(x[0].tolist()))


if __name__ == "__main__":
    main()
