# Assignment 7: RLAIF

CSCI 5942: AI Engineering, Fall 2026. Released Tue Nov 3, due Tue Nov 17
at 11:59 PM on Gradescope.

In A5 you fine-tuned GPT-2 medium to write restaurant descriptions from a
meaning representation, and the references told you how well it did. This
assignment trains that model further with the loop from lecture 19 (Oct
29), GRPO, with one change: the reward is a judge. Gemini (or your local LLM) reads a rubric
you write and scores each sampled description; the policy moves toward
what the judge likes; no reference is in sight. That is reinforcement
learning from AI feedback. Then you measure what the judge gained you (its
own score), what it cost (BLEU against references the judge never saw,
slot coverage, the FineWeb loss) and what the policy found in your rubric.

## What comes from where

| file | from | changed |
| :--- | :--- | :--- |
| `model.py` | A5 | nothing |
| `sft.py` | A5, with Part 2 filled in | nothing; the prompt template and `format_example` are what this assignment needs |
| `train_sft.py` | A5 `train.py` | the LoRA path and the memory report removed; rebuilds the A5 policy if you lost it |
| `judge.py` | A2 `judge.py` | A2's Gemini call and `--mock`, plus a local backend, a cache, and the group prompt; Parts 1 and 4 edit `RUBRICS` |
| `grpo.py` | lecture 19's GRPO results | Transformers replaced by your model class; the four ``fenced'' blocks are Part 2 |
| `eval.py` | A5 `eval.py` | one more column, the judge's score |
| `sample.py`, `plot_run.py`, `data/`, `slurm/` | A5 | given |

## Time

Estimated time to code each part by hand (no coding agents) building on A1-A5,
GPU time on one Delta A40 or course RTX 8000, and judge calls. A rerun of
anything hits the judge cache (`out/judge-cache.json`) and doesn't contribute
to the overall total (according to my calculations!).

| part | by hand | GPU | judge calls |
| :--- | ---: | ---: | ---: |
| 0 Environment | 30 min | 10 min if you rebuild the policy | none |
| 1 The judge | 1 h | none | about 60 |
| 2 The loop | 3 h | none | none |
| 3 Train | 45 min | 20 min | about 1,200 |
| 4 Leash and rubric | 1 h | 40 min | about 2,400 |
| The question | 1.5 h | | |

## Part 0: Environment (nothing to submit)

```bash
git clone git@github.com:csci5942/assignment-7.git
cd assignment-7
python3 -m venv .venv            # on Delta: module load pytorch-conda/2.12 first,
source .venv/bin/activate        # then add --system-site-packages to reuse its torch
pip install -r requirements.txt
python data/prepare_e2e.py       # 12 MB download, writes data/e2e/
ln -s ../../assignment-5/out/gpt2-medium-full out/sft     # your A5 policy
python check_env.py
python -m pytest tests/ -q       # every test fails with NotImplementedError until Part 2
```

If your A5 run is gone, `python train_sft.py` rebuilds it into `out/sft`
(A5's Part 3, one epoch, about three minutes on a 4090 and eight on an
A40).

### The judge: Gemini (or your local model, if you have access)

The judge is Gemini, as in A2, or a local model. Three things to set up once:

1. The key. The one you made for A2 works; if it is gone, make a new API
   key in Google AI Studio under the Google Cloud project that holds your
   course credit, and put it in the environment of every shell you run
   these scripts from:

   ```bash
   export GEMINI_API_KEY=...        # or GOOGLE_API_KEY; on Delta, in the sbatch script
   python judge.py --part1 --judge gemini --rubric given   # about 60 calls, a minute
   ```

2. The model is `gemini-2.5-flash` (`JUDGE_MODEL` in `judge.py`), called
   with the thinking budget at zero so a call returns in about a second.
   Every call is cached in `out/judge-cache.json` by the prompt's hash, so
   a rerun of any part, including a crashed run, costs nothing.

3. What it costs. A training run is about 800 calls (8 a step, each
   about 600 tokens in and 40 out); an `eval.py` row is 200. The whole
   assignment is about 3,000 calls, a few cents of credit at Flash
   prices (check the current price page before you start, and make sure
   you watch carefully; surprise billing may be a part of an AI engineer's
   life, but we don't want you to be hit with it as a student). If you hit a
   rate limit the scripts back off and retry; `JUDGE_PARALLEL=4` in the
   environment lowers the calls in flight.

Every script takes `--judge mock` for A2's deterministic fake scores, so
the plumbing can be tested for free; `--judge ollama:<model>` for a local
judge through Ollama; and `--judge openai:<model>` with `OPENAI_BASE_URL`
and `OPENAI_API_KEY` for any OpenAI-compatible server.

## Part 1: The judge

Before the judge trains anything, see how it grades people. For 30 test
MRs, `judge.py --part1` hands the judge a group of up to seven human
references for that MR plus one intruder, a reference written for a
different restaurant, in one call, the way `grpo.py` will hand it a group
of samples. Every group is scored twice, once in its order and once
reversed.

```bash
python judge.py --part1 --judge mock        # the plumbing
python judge.py --part1 --judge gemini      # about 60 calls
```

It prints how many of the scores moved by two or more points when the
order was reversed, how often the intruder scored lowest, and the mean
score the humans got. Read `results/judge-part1-given.json` for the
groups themselves. The rubric it used is `RUBRICS["given"]` in
`judge.py`; read that too, because Parts 3 and 4 train against it.

**Deliverable:** the three numbers, and two sentences: what the
order-swap number means for the advantage in Part 2 (the advantage is a
difference within a group), and whether the humans' mean score leaves the
policy anything to gain.

## Part 2: The loop

Lecture 19's loop, with your model class. Implement the four fenced
blocks in `grpo.py`; each has a test.

- `sample_group(model, prompt_ids, G, temperature, max_new, device)`:
  $G$ descriptions of one prompt, sampled token by token. A5's
  `sample.py` loop with `torch.multinomial` on the softmax of the logits
  divided by the temperature in place of `argmax`. All $G$ rows share the
  prompt, so one batch, no padding. A row stops at its first
  `<|endoftext|>`, which it keeps, or at `max_new` tokens.
- `group_advantage(rewards)`: $\hat A_i = (r_i - \text{mean}) / (\text{std} + 10^{-4})$
  over the group, as in DeepSeekMath and on the lecture 19 slide.
- `completion_logprobs(model, prompt_ids, comps, device)`: the
  teacher-forced $\log p$ of every completion token, for the $G$
  completions of one prompt. Right-pad with `<|endoftext|>` as A5's
  `collate` did, run one forward pass, and read the log-probability of
  each actual next token off the logits at the position before it. Return
  a `(G, T_max)` tensor and a mask that is 1 on real completion tokens.
- `grpo_loss(logp, ref_logp, mask, adv, beta)`: per token,
  $-\hat A_i \log \pi_\theta(o_{i,t}) + \beta\, k_3$ with
  $k_3 = e^{d} - d - 1$ and $d = \log \pi_{\text{ref}} - \log \pi_\theta$;
  the mean over each completion's tokens, then the mean over the group.
  Return the loss and the mean $k_3$.

```bash
python -m pytest tests/ -q
python sample.py --run out/sft --n 8       # a group at temperature 1, from your sampler
```

One update per batch means the policy that sampled the group is the
policy being updated, so the PPO ratio is 1 and the clip never fires; the
loss above is the whole algorithm. The reference model is a frozen bf16
copy of `out/sft`, and `beta` is the leash from lecture 19.

**Deliverable:** the four tests passing, the eight samples for one MR,
and one sentence on why a group whose scores are all equal contributes
nothing to the update.

## Part 3: Train

```bash
python grpo.py --config configs/judge-leash.json --judge mock --steps 2    # the plumbing
python grpo.py --config configs/judge-leash.json --judge gemini
python eval.py --run out/sft --judge gemini
python eval.py --run out/judge-leash --judge gemini
python plot_run.py out/judge-leash
```

100 steps of 8 MRs with $G = 8$, $\beta = 0.04$, about twenty minutes on
an A40 with the judge's eight calls a step in flight together. Each step
prints the group's mean score, the mean within-group spread (the signal
the advantage is made of), how many groups came back flat, the mean
length, the KL to the starting policy and the time split between
sampling, judging and the update. `out/judge-leash/samples.jsonl` keeps
the best and worst description of two groups every ten steps.

`eval.py` gives each policy one row. The judge's columns first: for each
of the first 200 test MRs the judge gets one group, as in training, seven
of the human references and your policy's description, shuffled, and
scores all eight in one call. That gives your policy's score on the
training reward's own scale with a 95% confidence interval, the humans'
mean score in the same groups, and your policy's rank among them (1 is
the best of eight). Then A5's columns, which the judge never saw: BLEU
against every reference, slot coverage, mean output tokens, and the
FineWeb loss.

**Deliverable:** the two rows (`out/sft` and `out/judge-leash`),
`curves.png`, two descriptions of the same MR before and after, and a
paragraph: what the judge score and the rank did, what BLEU and slot
coverage did, and whether they agree.

## Part 4: The leash and the rubric

Two more runs. First, drop the leash:

```bash
python grpo.py --config configs/judge-noleash.json --judge gemini     # beta = 0
python eval.py --run out/judge-noleash --judge gemini
```

Then read `samples.jsonl` from both runs and write down what the policy
found: what the highest-scored descriptions have in common that the
rubric did not ask for (length, a phrase, an attribute it invents, a
shape). That is your reward hacking, and Gao's curve from lecture 19 is
the reason. Write `RUBRICS["yours"]` in `judge.py` to close it, and run:

```bash
python grpo.py --config configs/judge-yours.json --judge gemini
python eval.py --run out/judge-yours --judge gemini                   # scored on the given rubric, like every row
python eval.py --run out/judge-yours --judge gemini --rubric yours    # and under its own
python plot_run.py out/judge-leash out/judge-noleash out/judge-yours
python plot_run.py --table results/sft.json results/judge-leash.json results/judge-noleash.json results/judge-yours.json
```

`eval.py` scores every row on the `given` rubric so the judge column is
one scale; the second `eval.py` line shows the policy under the rubric it
was trained on. A rubric is a judge prompt, so A2's Part 5 applies: a
paraphrase is a different reward. Keep `yours` about as long as `given`
and change what it asks for, not how it says it.

**Deliverable:** the four-row table, the curves, three descriptions from
`samples.jsonl` that show what the policy found, your rubric, and a
paragraph on what it changed.

## The question

Half a page at most: **what did the judge measure?** Cite the Part 1
numbers for the judge's own noise, the Part 3 rows for where the judge
and BLEU agree and where they do not, and the Part 4 runs for what the
rubric was actually rewarding. End with the number of judge calls, the
seconds per step the log reports, and what they would be for a checker.

In practice this runs on a current model with a serving engine for the
rollouts and one of the frameworks from the lecture 19 table; the loop
has the same four pieces.

## Deliverables

One PDF (Part 1 numbers and sentences; Part 2 tests and samples; Part 3
rows, `curves.png`, examples and paragraph; Part 4 table, curves,
samples, rubric and paragraph; the question) plus `grpo.py`, `judge.py`,
`out/*/log.jsonl`, `out/*/samples.jsonl` and `results/*.json`, via
Gradescope. Do not upload checkpoints.

## Layout

```
model.py             A5's model (do not edit)
sft.py               A5's template and mask, filled in (do not edit)
train_sft.py         rebuilds the A5 policy into out/sft, if needed
judge.py             the judge: backends, cache, prompts, RUBRICS (Parts 1 and 4 edit RUBRICS)
grpo.py              the loop; four fenced blocks (Part 2)
eval.py              judge score, BLEU, slot coverage, FineWeb loss (do not edit)
sample.py            greedy, or a group from your sampler
plot_run.py          curves.png and the results table
configs/             sft.json, judge-leash.json, judge-noleash.json, judge-yours.json
tests/               sampling, advantage, logprobs, loss
data/                prepare_e2e.py, fineweb_val.bin (A5's)
slurm/               A5's batch scripts, pointed at grpo.py and eval.py
out/, results/       written by the scripts (gitignored); out/judge-cache.json is the judge cache
```
