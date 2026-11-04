"""The judge: a model scoring restaurant descriptions against a rubric.

One call scores a whole group: the meaning representation, the G candidate
descriptions numbered, and the rubric go in; a JSON list of integers from
1 to 10 comes out. grpo.py uses those as the reward. eval.py scores one
description at a time with the same rubric, so a run's before and after
numbers sit on one scale. Part 1 runs the judge on the human references:

    python judge.py --part1 --judge gemini      # 30 test MRs, about 60 calls
    python judge.py --part1 --judge mock        # the plumbing, free

Backends (--judge): `gemini` is A2's call to gemini-2.5-flash with the
thinking budget set to zero (GEMINI_API_KEY or GOOGLE_API_KEY in the
environment, as in A2); `ollama:<model>` is a local model through Ollama's
/api/chat with thinking off; `openai:<model>` is any OpenAI-compatible
chat-completions server (OPENAI_BASE_URL and OPENAI_API_KEY in the
environment: vLLM, Ollama's /v1, a hosted provider); `mock` is A2's
deterministic hash, scores that mean nothing. Every reply is cached in
out/judge-cache.json under the prompt's sha256, so rerunning any part
costs nothing. JUDGE_PARALLEL in the environment caps the calls in flight
(default 8).

Part 1 and Part 4 edit RUBRICS. Nothing else in this file is edited.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
JUDGE_MODEL = "gemini-2.5-flash"      # A2's judge
JUDGE_THINKING = 0                    # thinking budget: no reasoning tokens, so each call is fast
OLLAMA = "http://127.0.0.1:11434"
CACHE = os.path.join(HERE, "out", "judge-cache.json")
PARALLEL = int(os.environ.get("JUDGE_PARALLEL", "8"))   # calls in flight at once
PART1_MRS = 30                        # test MRs Part 1 judges

RUBRICS = {
    # The rubric every graded run starts from.
    "given": (
        "The description states every attribute in the MR and nothing the MR "
        "does not say, in one or two sentences of natural English of the kind "
        "a restaurant guide would print."),
    # Part 4: your rubric, aimed at what the policy found in the one above.
    "yours": (
        "Replace this sentence with your rubric, about as long as the one above."),
}

GROUP_PROMPT = """\
You are judging short descriptions of a restaurant. Each one was written
from the meaning representation (MR) below, a list of attribute[value]
pairs.

Rubric: {rubric}

MR: {mr}

Candidates:
{candidates}

Score every candidate from 1 (useless) to 10 (as good as a careful human
writer), judging each on its own against the rubric and the MR, and use
the whole range. Reply with JSON only, no other text:
{{"scores": [<int>, <int>, ...]}} with one integer per candidate, in order.
"""

POINTWISE_PROMPT = """\
You are judging a short description of a restaurant, written from the
meaning representation (MR) below, a list of attribute[value] pairs.

Rubric: {rubric}

MR: {mr}

Candidate: {candidate}

Score the candidate from 1 (useless) to 10 (as good as a careful human
writer) against the rubric and the MR. Reply with JSON only, no other
text: {{"score": <int>}}
"""


# ------------------------------------------------------------------ backends

def ask_gemini(prompt, key, model=JUDGE_MODEL):
    """One call (A2's ask_gemini, plus the thinking budget). Returns the reply
    text and the token counts. Retries rate limits and server errors."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0,
                                            "responseMimeType": "application/json",
                                            "thinkingConfig": {"thinkingBudget": JUDGE_THINKING}}}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                reply = json.load(r)
            usage = reply.get("usageMetadata", {})
            text = reply["candidates"][0]["content"]["parts"][0]["text"]
            return text, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 503) or attempt == 5:
                raise
            time.sleep(2 ** attempt)


def ask_ollama(prompt, model):
    """One call to a local model, thinking off, JSON forced."""
    body = json.dumps({"model": model, "stream": False, "think": False, "format": "json",
                       "keep_alive": "10m",
                       "options": {"temperature": 0, "seed": 0, "num_predict": 256, "num_ctx": 4096},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        reply = json.load(r)
    return reply["message"]["content"], reply.get("prompt_eval_count", 0), reply.get("eval_count", 0)


def ask_openai(prompt, model):
    """One call to an OpenAI-compatible chat-completions server. The base
    URL ends in /v1; OPENAI_CHAT_TEMPLATE_KWARGS (JSON) is passed through
    for servers that take it, such as vLLM's {"enable_thinking": false}."""
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 256}
    extra = os.environ.get("OPENAI_CHAT_TEMPLATE_KWARGS")
    if extra:
        payload["chat_template_kwargs"] = json.loads(extra)
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=330) as r:
                reply = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or attempt == 5:
                raise
            time.sleep(2 ** attempt)
    usage = reply.get("usage") or {}
    return (reply["choices"][0]["message"]["content"] or "", usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0))


def mock_score(text):
    """A2's deterministic pseudo-score. Plumbing only."""
    h = hashlib.sha256(text.encode()).hexdigest()
    return 1 + int(h[:8], 16) % 10


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)       # JSON wrapped in prose
        if not m:
            raise ValueError(f"judge reply is not JSON: {text[:200]!r}")
        return json.loads(m.group(0))


# --------------------------------------------------------------------- judge

class Judge:
    """`spec` is gemini, ollama:<model>, or mock. Scores are floats 1 to 10."""

    def __init__(self, spec, rubric, cache_path=CACHE):
        self.spec = spec
        self.rubric = RUBRICS[rubric] if rubric in RUBRICS else rubric
        self.rubric_name = rubric
        self.backend, _, self.model = spec.partition(":")
        if self.backend == "gemini":
            self.model = self.model or JUDGE_MODEL
            self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not self.key:
                sys.exit("no GEMINI_API_KEY or GOOGLE_API_KEY in the environment; "
                         "use --judge mock to test the plumbing without a key")
        elif self.backend in ("ollama", "openai"):
            if not self.model:
                sys.exit(f"name the model: --judge {self.backend}:<model>")
            if self.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
                sys.exit("no OPENAI_API_KEY in the environment (and OPENAI_BASE_URL for a server other than OpenAI)")
        elif self.backend != "mock":
            sys.exit(f"unknown judge {spec!r}: gemini, ollama:<model>, openai:<model>, or mock")
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)
        self.lock = threading.Lock()
        self.calls = self.cache_hits = self.failures = 0
        self.prompt_tokens = self.output_tokens = 0

    def label(self):
        return "mock" if self.backend == "mock" else f"{self.backend}:{self.model}"

    def _ask(self, prompt):
        """The reply text for `prompt`, from the cache when it is there."""
        key = hashlib.sha256((self.label() + "\x00" + prompt).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                self.cache_hits += 1
                return self.cache[key]
        for attempt in range(4):           # a timeout or a dropped connection is retried, not fatal
            try:
                if self.backend == "gemini":
                    text, p, o = ask_gemini(prompt, self.key, self.model)
                elif self.backend == "openai":
                    text, p, o = ask_openai(prompt, self.model)
                else:
                    text, p, o = ask_ollama(prompt, self.model)
                break
            except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as e:
                if attempt == 3:
                    raise
                print(f"  judge: {type(e).__name__}, retrying in {5 * (attempt + 1)}s", flush=True)
                time.sleep(5 * (attempt + 1))
        with self.lock:
            self.calls += 1
            self.prompt_tokens += p
            self.output_tokens += o
            self.cache[key] = text
            n = self.calls
        if n % 50 == 0:
            self.save()                   # a crash mid-run keeps what was paid for
        return text

    def save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with self.lock:
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.cache, f)
            os.replace(tmp, self.cache_path)

    def score_group(self, mr, texts):
        """One call: a score per candidate. A reply that is not G integers is
        retried once, then the group gets 5 everywhere (zero advantage) and
        counts as a failure."""
        if self.backend == "mock":
            return [float(mock_score(t)) for t in texts]
        numbered = "\n".join(f"{i + 1}. {t if t else '(empty)'}" for i, t in enumerate(texts))
        prompt = GROUP_PROMPT.format(rubric=self.rubric, mr=mr, candidates=numbered)
        for attempt in range(2):
            try:
                scores = parse_json(self._ask(prompt))["scores"]
                if len(scores) == len(texts):
                    return [float(min(10, max(1, int(s)))) for s in scores]
            except (ValueError, KeyError, TypeError):
                pass
            prompt = prompt + "\n"       # a different cache key, a fresh call
        with self.lock:
            self.failures += 1
        return [5.0] * len(texts)

    def score_one(self, mr, text):
        if self.backend == "mock":
            return float(mock_score(text))
        prompt = POINTWISE_PROMPT.format(rubric=self.rubric, mr=mr, candidate=text if text else "(empty)")
        for attempt in range(2):
            try:
                return float(min(10, max(1, int(parse_json(self._ask(prompt))["score"]))))
            except (ValueError, KeyError, TypeError):
                prompt = prompt + "\n"
        with self.lock:
            self.failures += 1
        return 5.0

    def score_groups(self, groups):
        """[(mr, [text, ...]), ...] -> [[score, ...], ...], PARALLEL at a time."""
        with concurrent.futures.ThreadPoolExecutor(PARALLEL) as pool:
            out = list(pool.map(lambda g: self.score_group(*g), groups))
        self.save()
        return out

    def score_many(self, pairs):
        """[(mr, text), ...] -> [score, ...], pointwise, PARALLEL at a time."""
        with concurrent.futures.ThreadPoolExecutor(PARALLEL) as pool:
            out = list(pool.map(lambda p: self.score_one(*p), pairs))
        self.save()
        return out

    def stats(self):
        return {"judge": self.label(), "rubric": self.rubric_name, "calls": self.calls,
                "cache_hits": self.cache_hits, "failures": self.failures,
                "prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens}


# -------------------------------------------------------------------- Part 1

def part1(args):
    """The judge on the human references. For each of PART1_MRS test MRs the
    candidates are up to seven of its references plus one intruder, a
    reference written for a different MR. Every group is scored twice, in
    its order and reversed."""
    import sft
    rows = [r for r in sft.load_split("test") if len(r["refs"]) >= 4][:PART1_MRS]
    judge = Judge(args.judge, args.rubric)
    rng = random.Random(0)
    groups, intruder_at = [], []
    for i, r in enumerate(rows):
        other = rows[(i + PART1_MRS // 2) % len(rows)]["refs"][0]
        cands = r["refs"][:7] + [other]
        rng.shuffle(cands)
        groups.append((r["mr"], cands))
        intruder_at.append(cands.index(other))
    print(f"part 1: {len(groups)} groups of up to 8 on the '{args.rubric}' rubric ({judge.label()})")
    forward = judge.score_groups(groups)
    backward = judge.score_groups([(mr, list(reversed(c))) for mr, c in groups])
    backward = [list(reversed(b)) for b in backward]

    moved, total, intruder_lowest, intruder_scores, human_scores = 0, 0, 0, [], []
    for f, b, k in zip(forward, backward, intruder_at):
        for j, (x, y) in enumerate(zip(f, b)):
            total += 1
            moved += abs(x - y) >= 2
            (intruder_scores if j == k else human_scores).append((x + y) / 2)
        mean = [(x + y) / 2 for x, y in zip(f, b)]
        intruder_lowest += mean[k] <= min(mean)
    out = {"judge": judge.label(), "rubric": args.rubric, "groups": len(groups),
           "scores_total": total, "moved_2_or_more": moved, "moved_frac": round(moved / total, 3),
           "intruder_lowest": intruder_lowest, "intruder_lowest_frac": round(intruder_lowest / len(groups), 3),
           "human_mean": round(sum(human_scores) / len(human_scores), 2),
           "intruder_mean": round(sum(intruder_scores) / len(intruder_scores), 2),
           "stats": judge.stats(),
           "groups_detail": [{"mr": mr, "candidates": c, "forward": f, "backward": b, "intruder": k}
                             for (mr, c), f, b, k in zip(groups, forward, backward, intruder_at)]}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"judge-part1-{args.rubric}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  scores that moved by 2 or more when the order was reversed: {moved} of {total} ({100 * moved / total:.0f}%)")
    print(f"  groups where the intruder scored lowest: {intruder_lowest} of {len(groups)}")
    print(f"  mean score of the human references {out['human_mean']}, of the intruder {out['intruder_mean']}")
    print(f"  {judge.stats()}\n  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part1", action="store_true", help="judge the human references (Part 1)")
    ap.add_argument("--judge", default="gemini", help="gemini, ollama:<model>, openai:<model>, or mock")
    ap.add_argument("--rubric", default="given", choices=sorted(RUBRICS))
    args = ap.parse_args()
    if args.part1:
        part1(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
