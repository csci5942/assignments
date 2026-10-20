"""Generate from GPT-2 with and without your KV cache.

    python sample.py --prompt "The capital of France is" --tokens 100
    python sample.py --model gpt2-xl --prompt "..." "..." --tokens 200 --no-uncached

Runs greedy generation both ways (unless --no-uncached), checks that the
tokens agree, and prints each prompt's continuation and the time per
token. Several prompts run as one left-padded batch.
"""

import argparse
import time

import tiktoken
import torch

from model import from_pretrained, left_pad


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--prompt", nargs="+", default=["The capital of France is"])
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-uncached", action="store_true", help="skip the uncached run")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = from_pretrained(args.model).to(device)
    enc = tiktoken.get_encoding("gpt2")
    prompts = [enc.encode(p) for p in args.prompt]
    idx, pad = left_pad(prompts, pad_id=enc.eot_token)
    idx, pad = idx.to(device), pad.to(device)
    T = idx.size(1)

    model.generate(idx, 2, use_cache=True, pad=pad)            # warm up CUDA
    out, t = timed(lambda: model.generate(idx, args.tokens, args.temperature,
                                          use_cache=True, pad=pad))
    print(f"cached:   {t / args.tokens * 1e3:6.1f} ms/token")
    if not args.no_uncached and len(prompts) == 1:
        ref, t = timed(lambda: model.generate(idx, args.tokens, args.temperature))
        print(f"uncached: {t / args.tokens * 1e3:6.1f} ms/token")
        if args.temperature == 0:
            print("tokens match:", torch.equal(out, ref))
    for b, p in enumerate(args.prompt):
        print(f"\n--- {p!r}\n{p}{enc.decode(out[b, T:].tolist())}")


if __name__ == "__main__":
    main()
