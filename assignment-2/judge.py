"""LLM-judge scoring of eval.py samples, with an order-swap control.

Pointwise (one model, one rubric; grammar / coherence / style_match, each
1 to 10):

    python judge.py --results results/base.json --rubric shakespeare

Pairwise (two models, one rubric; every pair is judged twice with the
order swapped, and a verdict that flips on the swap counts as a tie):

    python judge.py --results results/base.json \
        --results-b results/base-tinystories.json --rubric tinystories

The judge is Gemini (GEMINI_API_KEY, or GOOGLE_API_KEY). --mock replaces
it with deterministic fake scores so the plumbing can be tested without
a key; mock numbers mean nothing.

With Opik configured, pointwise scores attach to the generation traces
eval.py created (their ids are in the results JSON) and each pairwise
comparison becomes its own trace. --no-opik skips that. Every run prints
a summary with 95% confidence intervals and writes a JSON to results/.

Part 5 edits two things in this file: RUBRICS and POINTWISE_PROMPT. An
edited prompt with an unchanged rubric writes the same filename and the
same Opik score names as the run before it, so pass --tag to keep the
two apart:

    python judge.py --results results/base.json --rubric shakespeare \
        --tag axes-reordered

Every output JSON records the prompt it was produced with (its text and
a sha256), and a run that would overwrite a file written under a
different prompt says so before it does.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
AXES = ("grammar", "coherence", "style_match")

RUBRICS = {
    "shakespeare": (
        "Elizabethan verse drama in the style of Shakespeare's plays: "
        "speaker headings in capitals, dialogue in short verse-like lines, "
        "archaic English (thou, hath, doth)."),
    "tinystories": (
        "A simple children's story in plain modern English: short "
        "sentences, small everyday vocabulary, named characters, a clear "
        "sequence of events."),
    "wikipedia": (
        "Encyclopedic prose in the register of English Wikipedia: "
        "declarative sentences, neutral tone, facts stated plainly."),
    "python-code": (
        "Python source code: def and class statements, docstrings, "
        "comments, plausible identifiers and indentation."),
}

POINTWISE_PROMPT = """\
You are grading a short text sample produced by a very small
character-level language model. It will be imperfect; grade it on its
own terms, using the full 1-10 range.

Target style: {rubric}

Score three axes, each an integer from 1 (complete failure) to 10
(indistinguishable from real text in the target style):
- grammar: well-formed words and sentences (or valid-looking code)
- coherence: stays on topic; each part follows from what came before
- style_match: closeness to the target style described above

Reply with JSON only, no other text:
{{"grammar": <int>, "coherence": <int>, "style_match": <int>, "comment": "<one sentence>"}}

SAMPLE:
<<<
{sample}
>>>
"""

PAIRWISE_PROMPT = """\
Two short text samples, A and B, each produced by a small
character-level language model. Both will be imperfect.

Target style: {rubric}

Which sample is better overall on grammar, coherence, and closeness to
the target style? Answer "tie" only if they are genuinely
indistinguishable in quality.

Reply with JSON only, no other text:
{{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}

SAMPLE A:
<<<
{a}
>>>

SAMPLE B:
<<<
{b}
>>>
"""


# ----------------------------------------------------------- provenance

def prompt_record(template, rubric_text):
    """The judge prompt this run used. Part 5 perturbs the
    prompt while the rubric name stays put, so the filename alone cannot
    tell two runs apart; this can."""
    text = template.replace("{rubric}", rubric_text)
    return {"sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
            "template": template, "rubric_text": rubric_text}


def warn_if_prompt_differs(out_path, record):
    """A run that replaces a file produced under a different prompt is
    almost always Part 5 without --tag. Say so; do not refuse, because a
    plain rerun after a failed judge call is legitimate."""
    if not os.path.exists(out_path):
        return
    try:
        with open(out_path, encoding="utf-8") as f:
            previous = json.load(f).get("judge_prompt", {}).get("sha256")
    except (OSError, ValueError):
        return
    if previous and previous != record["sha256"]:
        print(f"WARNING: results/{os.path.basename(out_path)} was written with a "
              f"different judge prompt ({previous}; this run is {record['sha256']}).\n"
              f"         This run will replace it. Stop now and pass --tag <label> "
              f"to keep both.")


# ------------------------------------------------------------------ judge

def ask_gemini(prompt, model, key):
    """One judge call. Returns the parsed JSON reply. Retries on rate
    limits and server errors with a short backoff."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0,
                                            "responseMimeType": "application/json"}}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                text = json.load(r)["candidates"][0]["content"]["parts"][0]["text"]
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 503) or attempt == 3:
                raise
            print(f"  gemini {e.code}, retrying in {2 ** (attempt + 1)}s")
            time.sleep(2 ** (attempt + 1))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)   # JSON wrapped in prose
        if not m:
            raise ValueError(f"judge reply is not JSON: {text[:200]!r}")
        return json.loads(m.group(0))


def mock_int(text, axis, lo=1, hi=10):
    """Deterministic pseudo-score from a hash. Plumbing only."""
    h = hashlib.sha256((axis + "\x00" + text).encode()).hexdigest()
    return lo + int(h[:8], 16) % (hi - lo + 1)


def score_one(text, rubric_text, args, key):
    """Pointwise: {grammar, coherence, style_match, comment}."""
    if args.mock:
        return {axis: mock_int(text, axis) for axis in AXES} | \
            {"comment": "mock score (deterministic hash, not a judgment)"}
    reply = ask_gemini(POINTWISE_PROMPT.format(rubric=rubric_text, sample=text), args.model, key)
    return {axis: int(reply[axis]) for axis in AXES} | {"comment": str(reply.get("comment", ""))}


def compare_one(a, b, rubric_text, args, key):
    """Pairwise, in the order given: 'A', 'B', or 'tie'."""
    if args.mock:
        return ("A", "B", "tie")[mock_int(a + "\x00" + b, "pairwise", 0, 2)]
    reply = ask_gemini(PAIRWISE_PROMPT.format(rubric=rubric_text, a=a, b=b), args.model, key)
    winner = str(reply.get("winner", "tie")).strip().upper()
    return winner if winner in ("A", "B") else "tie"


# ------------------------------------------------------------- statistics

def mean_ci(values):
    """Mean with a 95% confidence half-width (normal approximation)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"mean": mean, "n": n, "ci95": None}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return {"mean": round(mean, 3), "n": n, "ci95": round(1.96 * math.sqrt(var / n), 3)}


def winrate_ci(wins, n):
    """Win rate with a 95% confidence half-width. Approximate at n near 50."""
    if n == 0:
        return {"rate": None, "n": 0, "ci95": None}
    p = wins / n
    return {"rate": round(p, 3), "n": n, "ci95": round(1.96 * math.sqrt(p * (1 - p) / n), 3)}


# ------------------------------------------------------------------ modes

def run_pointwise(res, samples, rubric_text, args, key, client):
    name = res["model"]
    print(f"pointwise: {len(samples)} samples of '{name}' against the '{args.rubric}' "
          f"rubric{f' [{args.tag}]' if args.tag else ''} "
          f"({'MOCK' if args.mock else args.model})")
    scores = []
    for i, s in enumerate(samples):
        scores.append(score_one(s["text"], rubric_text, args, key) | {"index": s["index"]})
        if (i + 1) % 10 == 0:
            print(f"  judged {i + 1}/{len(samples)}")

    summary = {axis: mean_ci([sc[axis] for sc in scores]) for axis in AXES}
    for axis, st in summary.items():
        ci = f" +/- {st['ci95']}" if st["ci95"] is not None else ""
        print(f"  {axis:12s} {st['mean']}{ci}  (n={st['n']})")

    if client:
        # Attach each score to the generation's trace, so the traces view
        # shows every sample beside its grades.
        # The score name carries the tag too: without it a Part 5 rerun
        # would write the same names onto the same traces as Part 4.
        feedback = [{"id": s["trace_id"], "name": f"judge_{axis}_{args.label}",
                     "value": float(sc[axis]), "reason": sc["comment"]}
                    for s, sc in zip(samples, scores) if s.get("trace_id") for axis in AXES]
        if feedback:
            client.log_traces_feedback_scores(scores=feedback, project_name=res["opik"]["project"])
            client.flush()
            print(f"opik: attached {len(feedback)} feedback scores to '{name}' traces")
        else:
            print("opik: results JSON has no trace ids (eval.py ran with --no-opik), nothing to attach")

    return {"mode": "pointwise", "model": name, "scores": scores, "summary": summary}


def run_pairwise(res_a, res_b, samples_a, samples_b, rubric_text, args, key, client):
    name_a, name_b = res_a["model"], res_b["model"]
    n_pairs = min(len(samples_a), len(samples_b))
    print(f"pairwise: {n_pairs} pairs, '{name_a}' vs '{name_b}', rubric '{args.rubric}' "
          f"({'MOCK' if args.mock else args.model})")
    comparisons = []
    for i in range(n_pairs):
        ta, tb = samples_a[i]["text"], samples_b[i]["text"]
        # Ask twice, with A and B trading places, then name the winner.
        forward = {"A": name_a, "B": name_b, "tie": "tie"}[compare_one(ta, tb, rubric_text, args, key)]
        swapped = {"A": name_b, "B": name_a, "tie": "tie"}[compare_one(tb, ta, rubric_text, args, key)]
        flipped = forward != swapped
        comparisons.append({"index": i, "forward": forward, "swapped": swapped,
                            "flipped": flipped, "verdict": "tie" if flipped else forward})
        if (i + 1) % 10 == 0:
            print(f"  judged {i + 1}/{n_pairs} pairs")

    wins_a = sum(c["verdict"] == name_a for c in comparisons)
    wins_b = sum(c["verdict"] == name_b for c in comparisons)
    ties = n_pairs - wins_a - wins_b
    flips = sum(c["flipped"] for c in comparisons)
    summary = {
        "wins": {name_a: wins_a, name_b: wins_b, "tie": ties},
        "order_swap_flips": flips,
        "win_rate_excluding_ties": {name_a: winrate_ci(wins_a, wins_a + wins_b),
                                    name_b: winrate_ci(wins_b, wins_a + wins_b)},
    }
    print(f"  {name_a} {wins_a}, {name_b} {wins_b}, ties {ties} "
          f"({flips} verdicts flipped on the order swap and were counted as ties)")
    for nm, st in summary["win_rate_excluding_ties"].items():
        if st["rate"] is not None:
            print(f"  win rate {nm}: {st['rate']} +/- {st['ci95']} (n={st['n']}, ties excluded)")

    if client:
        for c in comparisons:
            client.trace(name="pairwise-judgment",
                         input={"rubric": args.rubric, "model_a": name_a, "model_b": name_b,
                                "index": c["index"]},
                         output={k: c[k] for k in ("verdict", "forward", "swapped", "flipped")},
                         tags=["a2-pairwise", args.label])
        client.flush()
        print(f"opik: logged {n_pairs} pairwise-judgment traces")

    return {"mode": "pairwise", "model_a": name_a, "model_b": name_b,
            "comparisons": comparisons, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="eval.py output JSON for model A")
    ap.add_argument("--results-b", default=None,
                    help="eval.py output JSON for model B (turns on pairwise mode)")
    ap.add_argument("--rubric", required=True, choices=sorted(RUBRICS))
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N samples")
    ap.add_argument("--tag", default=None,
                    help="label for this run, added to the output filename and the Opik "
                         "score names (Part 5: --tag paraphrase, --tag axes-reordered)")
    ap.add_argument("--mock", action="store_true", help="deterministic fake scores, no API key needed")
    ap.add_argument("--no-opik", action="store_true")
    args = ap.parse_args()
    if args.tag:
        args.tag = re.sub(r"[^A-Za-z0-9._-]+", "-", args.tag).strip("-")
    # Everything that names this run's output uses the rubric plus the tag.
    args.label = f"{args.rubric}-{args.tag}" if args.tag else args.rubric

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not args.mock and not key:
        sys.exit("no GEMINI_API_KEY or GOOGLE_API_KEY in the environment; "
                 "use --mock to test the plumbing without a key")

    with open(args.results, encoding="utf-8") as f:
        res_a = json.load(f)
    samples_a = res_a["samples"][:args.limit]
    rubric_text = RUBRICS[args.rubric]

    client = None
    if not args.no_opik:
        try:
            import opik
            client = opik.Opik(project_name=res_a["opik"].get("project") or "csci5942-a2")
        except Exception as e:
            print(f"opik: unavailable ({type(e).__name__}: {e}); "
                  f"scores will only be printed and written to JSON")

    res_b = None
    if args.results_b:
        with open(args.results_b, encoding="utf-8") as f:
            res_b = json.load(f)
        record = prompt_record(PAIRWISE_PROMPT, rubric_text)
        filename = f"judge-{res_a['model']}-vs-{res_b['model']}-{args.label}.json"
    else:
        record = prompt_record(POINTWISE_PROMPT, rubric_text)
        filename = f"judge-{res_a['model']}-{args.label}.json"

    # Named and checked before any judging, so a clobber is reported while
    # it can still be avoided.
    out_path = os.path.join(HERE, "results", filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    warn_if_prompt_differs(out_path, record)

    if res_b is not None:
        out = run_pairwise(res_a, res_b, samples_a, res_b["samples"][:args.limit],
                           rubric_text, args, key, client)
    else:
        out = run_pointwise(res_a, samples_a, rubric_text, args, key, client)

    out = {"rubric": args.rubric, "tag": args.tag,
           "judge_model": "mock" if args.mock else args.model,
           "mock": args.mock, "judge_prompt": record} | out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
