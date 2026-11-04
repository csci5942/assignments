"""Score a policy: the judge's own score, then what the judge never saw.

    python eval.py --run out/sft --judge gemini
    python eval.py --run out/judge-leash --judge gemini
    python eval.py --run out/sft --no-judge           # A5's columns only
    python eval.py --run out/judge-yours --judge gemini --rubric yours   # a policy under its own rubric

A5's eval.py (greedy generation per test MR, corpus BLEU against every
reference, slot coverage, forgetting as teacher-forced loss on held-out
FineWeb-Edu text) plus the judge's columns. For each of the first JUDGE_N
test MRs the judge gets one group, as in training: up to seven of the
human references and the policy's description, shuffled, one call. That
gives the policy's score on the training reward's own scale, the humans'
mean score in the same groups, and the policy's rank among them (1 is the
best of the group). Loads any run written by train_sft.py or grpo.py.
Not edited in Assignment 7.

Writes results/<name>.json.
"""

import argparse
import json
import math
import os
import random
import re
import time

import numpy as np
import sacrebleu
import torch

import judge as judgelib
import sft
from grpo import load_run

HERE = os.path.dirname(os.path.abspath(__file__))
FINEWEB = os.path.join(HERE, "data", "fineweb_val.bin")
MAX_NEW = 64
JUDGE_N = 200                 # test MRs the judge scores
JUDGE_GROUP = 8               # the policy's description plus up to seven human references
FINEWEB_BLOCKS = 200          # 200 x 1024 tokens, the same slice for everyone
FINEWEB_BATCH = 8
LITERAL_SLOTS = ("name", "near", "food", "eatType", "area")
SLOT_RE = re.compile(r"(\w+)\[(.*?)\]")


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


def slot_coverage(mr, text):
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


def mean_ci(values):
    """Mean with a 95% confidence half-width (A2's)."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return round(mean, 3), round(1.96 * math.sqrt(var / n), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="an out/<name> directory from train_sft.py or grpo.py")
    ap.add_argument("--judge", default="gemini", help="gemini, ollama:<model>, openai:<model>, or mock")
    ap.add_argument("--rubric", default="given", help="the rubric the judge scores with; 'given' for every row of a comparison table, 'yours' to see a policy under its own")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N meaning representations")
    ap.add_argument("--no-fineweb", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    model, _ = load_run(args.run, device)
    label = os.path.basename(os.path.normpath(args.run))
    with open(os.path.join(args.run, "summary.json")) as f:
        run_cfg = json.load(f)["config"]
    rubric = args.rubric
    rows = sft.load_split("test")
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
        "name": label, "rubric": rubric, "n_mrs": len(rows),
        "bleu": round(bleu.score, 2),
        "slot_coverage": round(hit / total, 4) if total else None,
        "mean_output_tokens": round(float(np.mean([len(sft.ENC.encode(h)) for h in hyps])), 1),
        "generation_seconds": round(gen_seconds, 1),
        "samples": samples,
    }
    if not args.no_judge:
        judge = judgelib.Judge(args.judge, rubric)
        n = min(JUDGE_N, len(rows))
        print(f"  judging {n} descriptions beside their human references ({judge.label()}, rubric '{rubric}')")
        rng = random.Random(0)
        groups, slots = [], []
        for r, h in zip(rows[:n], hyps[:n]):
            cands = r["refs"][:JUDGE_GROUP - 1] + [h]
            rng.shuffle(cands)
            groups.append((r["mr"], cands))
            slots.append(cands.index(h))
        scored = judge.score_groups(groups)
        scores, humans, ranks, beats = [], [], [], []
        for g, k in zip(scored, slots):
            mine = g[k]
            others = g[:k] + g[k + 1:]
            scores.append(mine)
            humans.append(sum(others) / len(others))
            ranks.append(1 + sum(1 for o in others if o > mine))
            beats.append(sum(1 for o in others if o < mine) / len(others))
        result["judge_mean"], result["judge_ci95"] = mean_ci(scores)
        result["human_mean"] = round(sum(humans) / n, 3)
        result["rank_mean"] = round(sum(ranks) / n, 2)
        result["beats_frac"] = round(sum(beats) / n, 3)
        result["judge_n"] = n
        result["judge"] = judge.stats()
        result["judge_scores"] = scores
        result["judge_ranks"] = ranks
        for s, sc, rk in zip(samples, scores, ranks):
            s["judge_score"], s["rank_of_8"] = sc, rk
    if not args.no_fineweb:
        result["fineweb_val_loss"] = round(fineweb_loss(model, device), 4)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"{label}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n{label}: "
          + (f"judge {result['judge_mean']} +/- {result['judge_ci95']} (humans {result['human_mean']}, rank {result['rank_mean']} of 8)  " if "judge_mean" in result else "")
          + f"BLEU {result['bleu']:.1f}  slot coverage {100 * result['slot_coverage']:.1f}%  "
          f"tokens {result['mean_output_tokens']}"
          + (f"  FineWeb loss {result['fineweb_val_loss']:.3f}" if "fineweb_val_loss" in result else "")
          + f"  ({len(rows)} MRs, {gen_seconds:.0f}s)")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
