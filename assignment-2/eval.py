"""Evaluate one A1-style checkpoint: loss on every corpus, plus samples.

    python eval.py --run ../assignment-1/out/base
    python eval.py --run ../assignment-1/out/base-tinystories --name base-tinystories
    python eval.py --run ../assignment-1/out/base --no-opik

For each corpus in corpora/ this computes teacher-forced nats/char under
the lecture protocol (protocol.py), then samples 50 generations from the
checkpoint. With Opik configured it logs one trace per generation, keeps
the corpora as a dataset, and records this model's loss row as an
experiment. With --no-opik (or when Opik is not installed or not
configured) it prints the same numbers and writes the same JSON; only
the dashboard is missing.

The JSON lands in results/<name>.json and is the input to judge.py.
"""

import argparse
import glob
import json
import os
import sys

import torch

import protocol

HERE = os.path.dirname(os.path.abspath(__file__))

# Sampling settings, fixed so everyone's samples are comparable.
SAMPLES = 50
NEW_CHARS = 300
TEMPERATURE = 0.8
TOP_K = 40
PROMPT = "\n"


def find_a1():
    """The A1 checkout: ../assignment-1 (the course repo name), or
    ../01-shakespeare (the instructor workspace)."""
    for name in ("assignment-1", "01-shakespeare"):
        path = os.path.normpath(os.path.join(HERE, "..", name))
        if os.path.isdir(path):
            return path
    return os.path.normpath(os.path.join(HERE, "..", "assignment-1"))


def load_model(run_dir, a1_dir, device):
    sys.path.insert(0, a1_dir)
    from model import GPT, GPTConfig  # noqa: E402
    ckpt = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    # torch.compile prefixes every parameter name; strip that back off.
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def load_corpora():
    paths = sorted(glob.glob(os.path.join(HERE, "corpora", "*.filtered.txt")))
    if not paths:
        sys.exit("no *.filtered.txt corpora found in corpora/")
    corpora = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            corpora[os.path.basename(p).removesuffix(".filtered.txt")] = f.read()
    return corpora


def generate_samples(model, stoi, itos, device):
    prompt = torch.tensor([[stoi[c] for c in PROMPT]], dtype=torch.long, device=device)
    samples = []
    for i in range(SAMPLES):
        torch.manual_seed(protocol.SEED + i)
        out = model.generate(prompt, NEW_CHARS, temperature=TEMPERATURE, top_k=TOP_K)
        samples.append({"index": i, "prompt": PROMPT, "seed": protocol.SEED + i,
                        "text": "".join(itos[t] for t in out[0].tolist())})
        if (i + 1) % 10 == 0:
            print(f"  sampled {i + 1}/{SAMPLES}")
    return samples


def log_to_opik(project, run_name, ckpt, losses, blocks, samples, corpora):
    """One trace per generation (so judge.py can attach scores to them), the
    corpora as a dataset, and this model's loss row as an experiment."""
    import opik
    from opik.evaluation import evaluate
    from opik.evaluation.metrics import base_metric, score_result

    client = opik.Opik(project_name=project)

    for s in samples:
        trace = client.trace(
            name="generation",
            input={"prompt": s["prompt"], "seed": s["seed"], "index": s["index"]},
            output={"text": s["text"]},
            metadata={"model": run_name, "n_params": ckpt["n_params"],
                      "temperature": TEMPERATURE, "top_k": TOP_K},
            tags=[run_name, "a2-generation"])
        s["trace_id"] = trace.id

    dataset = client.get_or_create_dataset(
        name="a2-eval-corpora",
        description="Lecture 4 corpora after normalization and OOV deletion")
    dataset.insert([{"corpus": c, "chars": len(t)} for c, t in sorted(corpora.items())])

    # The losses are already computed; the "task" only looks them up and
    # the metric reports them, which is all an experiment needs.
    class NatsPerChar(base_metric.BaseMetric):
        def __init__(self):
            super().__init__(name="nats_per_char")

        def score(self, output, **kwargs):
            return score_result.ScoreResult(
                name="nats_per_char", value=output["nats_per_char"],
                reason=f"{output['n_blocks']} blocks of {protocol.BLOCK} chars")

    def task(item):
        c = item["corpus"]
        return {"model": run_name, "corpus": c,
                "nats_per_char": losses[c], "n_blocks": blocks[c]}

    evaluate(dataset=dataset, task=task, scoring_metrics=[NatsPerChar()],
             experiment_name=f"{run_name}-loss-grid", project_name=project,
             experiment_config={"model": run_name, "n_params": ckpt["n_params"]},
             verbose=0)
    client.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing ckpt.pt")
    ap.add_argument("--name", default=None,
                    help="model name in the results (default: the run directory's name)")
    ap.add_argument("--a1-dir", default=find_a1(),
                    help="A1 checkout, for model.py and data/meta.json")
    ap.add_argument("--device", default="cpu",
                    help="the protocol numbers are defined on cpu; cuda matches "
                         "within noise and samples much faster")
    ap.add_argument("--project", default="csci5942-a2", help="Opik project name")
    ap.add_argument("--no-opik", action="store_true")
    args = ap.parse_args()

    run_name = args.name or os.path.basename(os.path.normpath(args.run))
    out_path = os.path.join(HERE, "results", f"{run_name}.json")

    with open(os.path.join(args.a1_dir, "data", "meta.json")) as f:
        itos = json.load(f)["itos"]
    stoi = {ch: i for i, ch in enumerate(itos)}

    model, ckpt = load_model(args.run, args.a1_dir, args.device)
    print(f"model {run_name}: {ckpt['n_params']:,} non-embedding params, "
          f"training val loss {ckpt['val_loss']:.4f}, device {args.device}")

    corpora = load_corpora()
    losses, blocks = {}, {}
    for name, text in sorted(corpora.items()):
        nats, n_blocks = protocol.nats_per_char(model, text, stoi, args.device)
        losses[name], blocks[name] = round(nats, 4), n_blocks
        print(f"  {name:12s} {nats:.4f} nats/char ({n_blocks} blocks)")

    print(f"sampling {SAMPLES} generations ({NEW_CHARS} chars each)")
    samples = generate_samples(model, stoi, itos, args.device)

    logged = False
    if not args.no_opik:
        try:
            log_to_opik(args.project, run_name, ckpt, losses, blocks, samples, corpora)
            logged = True
            print(f"opik: logged {SAMPLES} traces and the loss row to project '{args.project}'")
        except Exception as e:
            print(f"opik: logging failed ({type(e).__name__}: {e}); continuing without it. "
                  f"Run 'opik configure', or pass --no-opik to silence this.")

    result = {
        "model": run_name,
        "run_dir": os.path.abspath(args.run),
        "n_params": ckpt["n_params"],
        "train_best_val_loss": ckpt["val_loss"],
        "config": ckpt["config"],
        "protocol": {"block_size": protocol.BLOCK, "max_blocks_per_corpus": protocol.MAX_BLOCKS,
                     "seed": protocol.SEED, "device": args.device},
        "nats_per_char": losses,
        "blocks_used": blocks,
        "generation": {"prompt": PROMPT, "new_tokens": NEW_CHARS,
                       "temperature": TEMPERATURE, "top_k": TOP_K},
        "opik": {"logged": logged, "project": args.project if logged else None},
        "samples": samples,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
