# Assignment 2: Evaluation

CSCI 5942: AI Engineering, Fall 2026. Released Thu Sep 3, due Tue Sep 15
at 11:59 PM on Gradescope.

## What is Opik?

Opik is an open-source platform from Comet (Apache-2.0) for tracing and
evaluating LLM applications. The basic record is a *trace*: one model
call with its input, its output, and whatever metadata you attach. Traces
collect *feedback scores* (a judge's grade, a human thumbs-up, a loss),
they can be grouped into *datasets*, and a run of a model over a dataset
is an *experiment* that you can diff against the previous one. The
dashboard is where you read all of this; the Python SDK is how your code
writes to it.

We use it here because it is an example of evaluation as infrastructure,
seamlessly applying the same rubric on every checkpoint,
regressions showing up as diffs. In industry this job belongs to tools
like Opik, W&B Weave, LangSmith, or Arize Phoenix; Opik is the one that
self-hosts with a single Docker command and accepts any judge model,
including the course's Gemini, so it fits on the cloud VM you already
have or on your own laptop. In this assignment your generations become traces, the judge's
scores attach to them, and each model's loss row lands as an experiment.

Your A1 base run finished with a val loss near 1.47. This assignment
is about what that number does and does not tell you. You will
evaluate your A1 checkpoints on four different corpora, train an
identical twin of the base model on TinyStories, and put both models in
front of an LLM judge. Based on this we will then struggle with the
trade space of: which model is better, and on what exam?

Everything builds on your A1 checkpoints, but if you want to use our
version, we have a skeleton of that in `a1-skeleton.zip` (code only; the
whole ladder retrains in about five minutes on one GPU). The two
assignment checkouts are assumed to sit side by side
(`assignment-1/`, `assignment-2/`); pass `--a1-dir` to every script
if yours do not. If you like your version of A1, we encourage you
to use it!

## Part 0: Opik setup (nothing to submit)

The harness logs to Opik (Apache-2.0, github.com/comet-ml/opik), e.g.,
a trace per generation, a dataset of the corpora, an experiment per
model. Two ways to run it:

**Option A: self-hosted (needs Docker).**

This is the path I chose, which can be pretty easily done in a click-through
fashion with your favorite coding assistant:

```bash
git clone https://github.com/comet-ml/opik.git
cd opik && ./opik.sh
```

The dashboard comes up at http://localhost:5173. Point the SDK at it:

```bash
pip install opik
opik configure --use-local
```

**Option B: Comet free tier (no Docker).**

Create a free account at comet.com, open Opik, and copy your API key
from the user menu. The free quota (25k spans per month at the time of
writing) covers this assignment and any that build on it within this course.

```bash
pip install opik
opik configure          # paste the API key when prompted
```

Either way, `opik healthcheck` should be green before proceeding to Part 1.

## Part 1: Wire in your base checkpoint

```bash
python eval.py --run ../assignment-1/out/base
```

This computes teacher-forced loss in nats/char on the four corpora in
`corpora/` (as discussed in lecture, the process is: normalize, delete OOV
chars, score non-overlapping 256-char blocks; see `protocol.py` and
`corpora/README.md` for details), samples 50 generations, and logs everything
to Opik. The numbers and samples output to `results/base.json`.

Sanity check: the shakespeare number should be within a few
hundredths of your A1 best val loss. Ours is 1.43 against a training
val loss of 1.46 (training eval used random crops of the same split;
this pass uses non-overlapping blocks). If yours is off by tenths,
something is wrong and you should check to make sure there's no bug crawling
around (e.g., sample generations, OOV char deletion operator).

Deliverable: the four numbers (base's nats/char on shakespeare,
tinystories, wikipedia, and python-code), plus a screenshot of the Opik
traces view showing your 50 generations.

## Part 2: The grid

Run eval.py for every rung of your A1 ladder:

```bash
for m in pico nano micro mini base; do
  python eval.py --run ../assignment-1/out/$m
done
```

Plot the 5x4 result as a heatmap (models on one axis, corpora on the
other, nats/char in the cells; matplotlib, any style you want).
Report the OOV fractions from `corpora/README.md` next to it: the
python-code column lost 7.7% of its characters in L04, so yours will probably
be close-ish.

Deliverable: the heatmap, the OOV fractions, and two sentences on the
python-code column. Only the shakespeare column improves monotonically
with model size, and python-code is the worst regression: our
27k-parameter pico outscores every rung except base. What is a bigger
Shakespeare model achieving, and why does it not transfer here?

## Part 3: The twin

Train the (original unaltered) base config on TinyStories ("the twin"). Before
anything else, write down two predictions: the twin's nats/char on tinystories
and on shakespeare.

```bash
python prepare_tinystories.py
```

This downloads TinyStories and builds `data-tinystories/` with the
same 82M-char training budget A1 spent (5000 iters x 64 batch x 256
block), encoded with the A1 vocabulary under the same OOV protocol.
(Also please read the above python script for prep decisions!).
Note the asymmetry here: A1 cycled a 1.1M-char corpus for about 74 epochs,
while the twin sees 82M unique chars about once.

A1's train.py reads the `data/` directory next to itself, so swap the
twin data in for the run:

```bash
cd ../assignment-1
mv data data.shakespeare
ln -s ../assignment-2/data-tinystories data
python train.py --config configs/base.json --out-dir out/base-tinystories
rm data && mv data.shakespeare data
cd ../assignment-2
```

Training takes about the same amount of time your A1 base run did (~2 minutes).
Then give the twin its own row:

```bash
python eval.py --run ../assignment-1/out/base-tinystories --name base-tinystories
```

Deliverable: the twin's train/val curves, its row appended to your
Part 2 heatmap, and predicted vs measured for your two numbers, with
one sentence on the larger miss.

## Part 4: The judge

The grid ranks models by teacher-forced loss. It cannot say whether
either model writes _anything a reader would accept_; for that you need
an exam with no answer key, and a judge.

`judge.py` sends samples to Gemini (`GEMINI_API_KEY` or
`GOOGLE_API_KEY` in the environment; use your course GCP credits;
`--model` overrides the default gemini-2.5-flash) and scores each one
1 to 10 on grammar, coherence, and style_match against a per-corpus
rubric. Test the plumbing first without spending anything:

```bash
python judge.py --results results/base.json --rubric shakespeare --mock
```

`--mock` produces deterministic fake scores that mean nothing;
`--limit 5` is the cheap way to try a real call.

For the real runs, we will run both models on both rubrics:

```bash
python judge.py --results results/base.json --rubric shakespeare
python judge.py --results results/base.json --rubric tinystories
python judge.py --results results/base-tinystories.json --rubric shakespeare
python judge.py --results results/base-tinystories.json --rubric tinystories
```

Finally, we'll do a head-to-head, where every pair is judged twice with the
order swapped. This is to make sure that we aren't implicitly learning position
bias!  If A and B switch places in an eval that points to having been corrupted
by that bias. Note, if this happens, judge.py counts it as a tie and reports how often
it happened.

```bash
python judge.py --results results/base.json \
    --results-b results/base-tinystories.json --rubric shakespeare
python judge.py --results results/base.json \
    --results-b results/base-tinystories.json --rubric tinystories
```

Pointwise scores populate the generation traces in Opik, so the
`traces` view now shows every sample with its judge scores. Pairwise
verdicts get their own traces.

Deliverable: the table of pointwise means with 95% confidence
intervals (judge.py prints them), the two pairwise win rates with CIs
and the flip counts, and a screenshot of the scored traces.

## Part 5: Perturbing the judge

The judge is model, so it is also an eval target. Perturb its prompt twice and
measure how much your Part 4 numbers move.

1. **Paraphrase the rubric.** Add a `shakespeare_v2` entry to the
   `RUBRICS` dict in judge.py: same meaning as `shakespeare`, none of
   the same phrasing! Rerun the pointwise scoring of `results/base.json`
   against it.
2. **Reorder the axes.** In `POINTWISE_PROMPT`, move style_match first
   and grammar last. Rerun the original `shakespeare` rubric.

Both runs score the same 50 samples as Part 4, so differences are only as a
result of the prompt. Put the three pointwise means side by side with their
CIs.

Deliverable: the comparison table and a two-sentence verdict: do your
Part 4 conclusions survive a paraphrase, and if not, which claims were
inside the noise? (Lecture 4 showed leaderboard ranks moving under
choice reordering; this is the same experiment run on your own judge.)

## The question

Answer in your report, half a page at most: **which model is better,
and on what exam?** Cite your grid for the loss claim and your judge
tables for the quality claim, and point at where the two disagree.

## Deliverables

One PDF (heatmap + OOV table + your two sentences, twin curves +
predictions, judge tables, the judge-perturbation table and verdict,
the two screenshots, your answer to the question) plus your
`results/*.json`, via Gradescope.

## Layout

```
protocol.py               the lecture eval protocol (do not edit)
corpora/                  the four exams + provenance (do not edit)
eval.py                   loss row + 50 samples + Opik logging
judge.py                  Gemini judge: rubric scores, order-swap control
prepare_tinystories.py    builds data-tinystories/ for the twin
results/                  eval.py and judge.py output (gitignored)
```
