"""Score a fine-tuned model: BLEU and slot coverage on E2E, and what it
forgot, as teacher-forced loss on held-out FineWeb-Edu text.

    python eval.py --run out/gpt2-medium-full
    python eval.py --base gpt2-medium            # the untouched model's row

Greedy generation per prompt, stopping at <|endoftext|> or MAX_NEW
tokens. BLEU: sacrebleu corpus BLEU against every reference of each
meaning representation. Slot coverage: of the slots with a literal string
value (name, near, food, eatType, area), the fraction that appear verbatim
in the output. Forgetting: A2's protocol on A4's data, mean loss over
fixed 1024-token blocks of data/fineweb_val.bin in eval mode, nats per
token. Not edited in Assignment 5.

Writes results/<name>.json.
"""

import argparse
import json
import os
import re
import time

import numpy as np
import sacrebleu
import torch

import lora
import sft
from model import GPT, GPTConfig, from_pretrained, gpt2_config

HERE = os.path.dirname(os.path.abspath(__file__))
FINEWEB = os.path.join(HERE, "data", "fineweb_val.bin")
MAX_NEW = 64
FINEWEB_BLOCKS = 200          # 200 x 1024 tokens, the same slice for everyone
FINEWEB_BATCH = 8
LITERAL_SLOTS = ("name", "near", "food", "eatType", "area")
SLOT_RE = re.compile(r"(\w+)\[(.*?)\]")


def load(args, device):
    if args.base:
        model, label = from_pretrained(args.base), args.base
    else:
        with open(os.path.join(args.run, "summary.json")) as f:
            cfg = json.load(f)["config"]
        model = from_pretrained(cfg["model"])
        if cfg["mode"] == "lora":
            lora.apply_lora(model, cfg["rank"], cfg["alpha"], tuple(cfg["targets"]))
            state = torch.load(os.path.join(args.run, "lora.pt"), map_location="cpu")["lora"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            assert not unexpected, unexpected
            lora.merge_lora(model)           # Part 4: the served model is plain GPT-2 again
        else:
            state = torch.load(os.path.join(args.run, "ckpt.pt"), map_location="cpu")["model"]
            model.load_state_dict(state)
        label = os.path.basename(os.path.normpath(args.run))
    return model.to(device).eval(), label


@torch.no_grad()
def generate(model, prompt_ids, device):
    """Greedy continuation of one prompt until EOT or MAX_NEW tokens."""
    ids = torch.tensor([prompt_ids], device=device)
    out = []
    for _ in range(MAX_NEW):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, _ = model(ids[:, -model.config.block_size:])
        nxt = logits[0, -1].argmax().item()
        if nxt == sft.EOT:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    return sft.ENC.decode(out).strip()


def slot_coverage(mr: str, text: str):
    slots = [(k, v) for k, v in SLOT_RE.findall(mr) if k in LITERAL_SLOTS]
    hit = sum(1 for _, v in slots if v.lower() in text.lower())
    return hit, len(slots)


@torch.no_grad()
def fineweb_loss(model, device):
    data = np.fromfile(FINEWEB, dtype=np.uint16)
    T = model.config.block_size
    n = min(FINEWEB_BLOCKS, (len(data) - 1) // T)
    total = 0.0
    for i in range(0, n, FINEWEB_BATCH):
        rows = [torch.from_numpy(data[j * T:(j + 1) * T + 1].astype(np.int64))
                for j in range(i, min(n, i + FINEWEB_BATCH))]
        xy = torch.stack(rows).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(xy[:, :-1], xy[:, 1:])
        total += loss.item() * len(rows)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="an out/<name> directory from train.py")
    g.add_argument("--base", help="a GPT-2 size, evaluated with no fine-tuning")
    ap.add_argument("--split", default="test", choices=("test", "dev"))
    ap.add_argument("--limit", type=int, default=None, help="score only the first N meaning representations")
    ap.add_argument("--no-fineweb", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    model, label = load(args, device)
    rows = sft.load_split(args.split)
    if args.split == "dev":                      # dev is pair-per-row; group it like test
        grouped = {}
        for r in rows:
            grouped.setdefault(r["mr"], []).append(r["ref"])
        rows = [{"mr": mr, "refs": refs} for mr, refs in grouped.items()]
    if args.limit:
        rows = rows[:args.limit]

    t0 = time.time()
    hyps, samples, hit, total = [], [], 0, 0
    for i, r in enumerate(rows):
        prompt_ids, _ = sft.format_example(r["mr"], "")
        hyp = generate(model, prompt_ids, device)
        hyps.append(hyp)
        h, t = slot_coverage(r["mr"], hyp)
        hit += h
        total += t
        if i < 5:
            samples.append({"mr": r["mr"], "output": hyp, "reference": r["refs"][0]})
        if i % 100 == 0:
            print(f"  {i}/{len(rows)} generated, {time.time() - t0:.0f}s")
    gen_seconds = time.time() - t0

    # sacrebleu wants the same number of references for every hypothesis;
    # repeating a reference changes nothing in multi-reference BLEU.
    k = max(len(r["refs"]) for r in rows)
    ref_sets = [[(r["refs"] * k)[j] for r in rows] for j in range(k)]
    bleu = sacrebleu.corpus_bleu(hyps, ref_sets)

    result = {
        "name": label, "split": args.split, "n_mrs": len(rows),
        "bleu": bleu.score, "bleu_signature": str(bleu.format(signature=True)).split("\n")[-1] if hasattr(bleu, "format") else None,
        "slot_coverage": hit / total if total else None,
        "mean_output_tokens": float(np.mean([len(sft.ENC.encode(h)) for h in hyps])),
        "generation_seconds": gen_seconds,
        "samples": samples,
    }
    if not args.no_fineweb:
        if not os.path.exists(FINEWEB):
            raise FileNotFoundError(f"{FINEWEB} missing; it ships with the repo")
        result["fineweb_val_loss"] = fineweb_loss(model, device)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"{label}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n{label}: BLEU {bleu.score:.1f}  slot coverage {100 * result['slot_coverage']:.1f}%"
          + (f"  FineWeb val loss {result['fineweb_val_loss']:.3f}" if "fineweb_val_loss" in result else "")
          + f"  ({len(rows)} MRs, {gen_seconds:.0f}s)")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
