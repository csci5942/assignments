"""Write a description from a run's policy.

    python sample.py --run out/sft --mr "name[The Vaults], eatType[pub], near[Café Adriatic]"
    python sample.py --run out/judge-leash --mr "..."
    python sample.py --run out/sft --mr "..." --n 8          # a group at temperature 1 (Part 2)

Greedy by default, as eval.py. With --n, the group sampler you wrote in
grpo.py. Given.
"""

import argparse

import torch

import sft
from grpo import decode, load_run, sample_group

DEFAULT_MR = "name[The Vaults], eatType[pub], priceRange[more than £30], customer rating[5 out of 5], near[Café Adriatic]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="an out/<name> directory")
    ap.add_argument("--mr", default=DEFAULT_MR, help="an E2E meaning representation")
    ap.add_argument("--n", type=int, default=1, help="sample this many at --temperature instead of greedy")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, _ = load_run(args.run, args.device)
    ids, _ = sft.format_example(args.mr, "")
    if args.n > 1:
        for comp in sample_group(model, ids, args.n, args.temperature, 64, args.device):
            print(decode(comp))
        return
    x = torch.tensor([ids], device=args.device)
    with torch.no_grad():
        for _ in range(64):
            logits, _ = model(x)
            nxt = logits[0, -1].argmax().item()
            if nxt == sft.EOT:
                break
            x = torch.cat([x, torch.tensor([[nxt]], device=args.device)], dim=1)
    print(sft.ENC.decode(x[0, len(ids):].tolist()).strip())


if __name__ == "__main__":
    main()
