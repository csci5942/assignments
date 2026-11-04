"""GRPO on the Assignment 5 policy, with a judge for the reward.

    python grpo.py --config configs/judge-leash.json --judge gemini
    python grpo.py --config configs/judge-leash.json --judge mock --steps 2    # the plumbing

Lecture 19's loop, with the model class you wrote in place of Transformers.
Each step takes `prompts_per_step` meaning representations from E2E train,
samples `group` descriptions of each at `temperature` (Part 2,
sample_group), asks the judge for a score per description (judge.py, one
call per group), turns the scores into group-relative advantages (Part 2,
group_advantage) and takes one AdamW step on the GRPO loss (Part 2,
grpo_loss): per token, minus the advantage times the log-probability
(Part 2, completion_logprobs), plus beta times the k3 estimate of the KL to
the frozen starting policy. One update per batch, so the policy that
sampled is the policy being updated: the PPO ratio is 1 and the clip never
fires, as in lecture 19.

Precision as in A4 and A5: fp32 weights, bf16 autocast, the reference copy
in bf16. GPT-2 medium at G = 8 and 64 new tokens needs about 9 GiB.

Writes out/<name>/log.jsonl (one line per step), samples.jsonl (the best
and worst description of two groups every SAMPLE_EVERY steps),
summary.json and ckpt.pt, in A5's layout so eval.py and sample.py read it.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

import judge as judgelib
import sft
from model import GPT, gpt2_config

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_EVERY = 10


def load_run(run_dir, device, dtype=torch.float32):
    """A5's checkpoint layout: summary.json names the size, ckpt.pt holds
    the weights. The same loader reads out/sft and any out/<grpo run>."""
    with open(os.path.join(run_dir, "summary.json")) as f:
        cfg = json.load(f)["config"]
    model = GPT(gpt2_config(cfg["model"]))
    state = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu")["model"]
    model.load_state_dict(state)
    return model.to(device=device, dtype=dtype).eval(), cfg["model"]


# ------------------------------------------------------------------- Part 2

@torch.no_grad()
def sample_group(model, prompt_ids, G, temperature, max_new, device):
    """G completions of one prompt, sampled token by token at `temperature`
    (A5's sample.py loop with multinomial in place of argmax). Returns a
    list of G token lists; each stops at the first EOT, which is kept, or
    at max_new tokens."""
    # --- YOUR IMPLEMENTATION HERE ---
    raise NotImplementedError("implement this block")


def group_advantage(rewards):
    """DeepSeekMath's outcome advantage: (r - mean of the group) / (std of
    the group + 1e-4), one float per completion."""
    # --- YOUR IMPLEMENTATION HERE ---
    raise NotImplementedError("implement this block")


def completion_logprobs(model, prompt_ids, comps, device):
    """Teacher-forced log p of every completion token under `model`, for G
    completions of one prompt. Returns (logp, mask), two (G, Tmax) float32
    tensors; mask is 1 on real completion tokens and 0 on the EOT padding
    that fills shorter rows (A5's collate, right-padded with EOT)."""
    # --- YOUR IMPLEMENTATION HERE ---
    raise NotImplementedError("implement this block")


def grpo_loss(logp, ref_logp, mask, adv, beta):
    """The loss for one group. Per token: -A * log p + beta * k3, with
    k3 = exp(ref - logp) - (ref - logp) - 1 (DeepSeekMath's KL estimate);
    the mean over each completion's tokens, then the mean over the group.
    `adv` is a (G,) tensor. Returns (loss, mean k3 per token)."""
    # --- YOUR IMPLEMENTATION HERE ---
    raise NotImplementedError("implement this block")


# --------------------------------------------------------------------- loop

def decode(comp):
    return sft.ENC.decode(comp[:-1] if comp and comp[-1] == sft.EOT else comp).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--judge", default="gemini", help="gemini, ollama:<model>, openai:<model>, or mock")
    ap.add_argument("--steps", type=int, default=None, help="override the config")
    ap.add_argument("--out", default=None, help="default out/<name>")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    if args.steps:
        cfg["steps"] = args.steps
    out_dir = args.out or os.path.join(HERE, "out", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    device = args.device
    G, n_prompts = cfg["group"], cfg["prompts_per_step"]
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # ---- the policy, its frozen twin, the optimizer
    init = cfg["init"] if os.path.isabs(cfg["init"]) else os.path.join(HERE, cfg["init"])
    policy, size = load_run(init, device)
    ref, _ = load_run(init, device, dtype=torch.bfloat16)
    for p in ref.parameters():
        p.requires_grad_(False)
    for p in policy.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0.0)

    # ---- the prompts: one of each training MR, shuffled
    seen, mrs = set(), []
    for r in sft.load_split("train"):
        if r["mr"] not in seen:
            seen.add(r["mr"])
            mrs.append(r["mr"])
    random.Random(cfg["seed"]).shuffle(mrs)
    judge = judgelib.Judge(args.judge, cfg["rubric"])
    print(f"{cfg['name']}: {size} from {cfg['init']}, {cfg['steps']} steps x {n_prompts} prompts x G={G}, "
          f"beta {cfg['beta']}, lr {cfg['lr']}, judge {judge.label()} on the '{cfg['rubric']}' rubric")

    log = open(os.path.join(out_dir, "log.jsonl"), "w")
    samples = open(os.path.join(out_dir, "samples.jsonl"), "w")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for step in range(cfg["steps"]):
        t0 = time.time()
        batch = [mrs[(step * n_prompts + j) % len(mrs)] for j in range(n_prompts)]
        prompts = [sft.format_example(mr, "")[0] for mr in batch]

        # sample G descriptions per MR
        comps = [sample_group(policy, p, G, cfg["temperature"], cfg["max_new"], device) for p in prompts]
        texts = [[decode(c) for c in group] for group in comps]
        t_gen = time.time() - t0

        # the judge scores every group, one call each
        scores = judge.score_groups(list(zip(batch, texts)))
        t_judge = time.time() - t0 - t_gen

        # one update over all groups
        policy.train()
        opt.zero_grad(set_to_none=True)
        kls, losses = [], []
        for p_ids, group, s in zip(prompts, comps, scores):
            adv = torch.tensor(group_advantage(s), device=device, dtype=torch.float32)
            with torch.no_grad():
                ref_lp, _ = completion_logprobs(ref, p_ids, group, device)
            lp, mask = completion_logprobs(policy, p_ids, group, device)
            loss, kl = grpo_loss(lp, ref_lp, mask, adv, cfg["beta"])
            (loss / n_prompts).backward()
            losses.append(loss.item())
            kls.append(kl.item())
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
        opt.step()
        policy.eval()
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.time() - t0

        # the numbers
        flat = [x for s in scores for x in s]
        lens = [len(c) for g in comps for c in g]
        rec = {"step": step, "mean_reward": round(float(np.mean(flat)), 3),
               "group_std": round(float(np.mean([np.std(s) for s in scores])), 3),
               "groups_all_same": int(sum(1 for s in scores if np.std(s) == 0)),
               "mean_len": round(float(np.mean(lens)), 1),
               "frac_max_len": round(float(np.mean([len(c) >= cfg["max_new"] for g in comps for c in g])), 3),
               "mean_kl": round(float(np.mean(kls)), 6), "loss": round(float(np.mean(losses)), 6),
               "grad_norm": round(gnorm, 4), "gen_seconds": round(t_gen, 1),
               "judge_seconds": round(t_judge, 1), "seconds": round(seconds, 1),
               "judge_calls": judge.calls, "judge_cache_hits": judge.cache_hits, "judge_failures": judge.failures,
               "gpu_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device == "cuda" else 0}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"  step {step:3d}  reward {rec['mean_reward']:.2f} (group std {rec['group_std']:.2f}, "
              f"{rec['groups_all_same']} flat)  len {rec['mean_len']:5.1f}  kl {rec['mean_kl']:.4f}  "
              f"gn {gnorm:.2f}  {seconds:.0f}s (gen {t_gen:.0f}, judge {t_judge:.0f})")
        if step % SAMPLE_EVERY == 0 or step == cfg["steps"] - 1:
            for mr, group, s in list(zip(batch, texts, scores))[:2]:
                hi, lo = int(np.argmax(s)), int(np.argmin(s))
                samples.write(json.dumps({"step": step, "mr": mr, "best": group[hi], "best_score": s[hi],
                                          "worst": group[lo], "worst_score": s[lo]}) + "\n")
            samples.flush()
    log.close()
    samples.close()

    # ---- what eval.py and sample.py read
    summary = {
        "name": os.path.basename(os.path.normpath(out_dir)),
        "config": {"model": size, "mode": "full", **cfg},
        "judge": judge.stats(), "steps": cfg["steps"], "seconds": round(time.time() - t_start, 1),
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device == "cuda" else 0,
        "setup": "fp32 weights, bf16 autocast, bf16 reference; AdamW betas 0.9/0.99, no weight decay, "
                 "grad clip 1.0; one update per batch (ratio 1, no clip); advantage (r - mean) / (std + 1e-4); "
                 "loss per token -A log p + beta k3, mean over tokens then over the group; "
                 "sampling at the config temperature with no top-k or top-p, stop at EOT or max_new",
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({"model": policy.state_dict(), "config": summary["config"]}, os.path.join(out_dir, "ckpt.pt"))
    print(f"\n  done: {summary['seconds']:.0f}s, peak {summary['peak_gib']} GiB, judge {judge.stats()}")
    print(f"  -> {out_dir}/summary.json, ckpt.pt, log.jsonl, samples.jsonl")


if __name__ == "__main__":
    main()
