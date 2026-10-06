# Assignment 5: PEFT vs. FFT

CSCI 5942: AI Engineering, Fall 2026. Released Tue Oct 6, due Tue Oct 20
at 11:59 PM on Gradescope.

In A1 you wrote a decoder-only transformer and in A4 you widened it to
GPT-2's shapes. This assignment loads OpenAI's released GPT-2 weights
into that class and adapts the model to a new task two ways: full
fine-tuning (FFT, every weight updated) and LoRA (small low-rank adapters
beside frozen weights; the required reading). The task is E2E NLG, the
one the LoRA paper used for GPT-2: turn a list of restaurant attributes
into a sentence. You will measure what each method takes in memory and
time, what it gains on the task, and what it forgets. The question at the
end: when is updating every weight worth it?

## What comes from where

| file | from | changed |
| :--- | :--- | :--- |
| `model.py` | A4 `trainer/model.py` | nothing, plus `from_pretrained()` at the bottom (given). A1's layer names are GPT-2's, so the loader is forty lines. |
| `train.py` | A4 `trainer/train.py`, single-GPU path | batches come from your `collate` (Part 2); `mode: lora` calls your `apply_lora` (Part 4); A4's memory report at the end |
| `sft.py` | new | the template and the loss mask (Part 2) |
| `lora.py` | new | `LoRALinear`, `apply_lora`, `merge_lora` (Part 4) |
| `eval.py` | A2 `protocol.py` | greedy generation and BLEU on E2E, plus A2's teacher-forced loss on A4's FineWeb-Edu validation data (given) |
| `data/prepare_e2e.py`, `sample.py`, `plot_knee.py`, `slurm/` | A2, A1, A4 | given |

The tokenizer is A4's (GPT-2 BPE, tiktoken). The "params + grads +
optimizer" column of the A4 README is Part 1's arithmetic.

## Time

Estimated time to code each part by hand (no assistant) after A1 to A4,
and GPU time on one Delta A40 or course RTX 8000. Under two GPU-hours
total.

| part | by hand | GPU |
| :--- | ---: | ---: |
| 0 Environment | 30 min | none |
| 1 Memory model | 1 h arithmetic, 30 min runs | 10 min |
| 2 Template and mask | 1.5 to 2 h | none |
| 3 Full fine-tuning | 45 min | 15 min |
| 4 LoRA and the rank knee | 2 h | 30 min |
| 5 Which matrices | 45 min | 15 min |
| Report | 1.5 h | |

## Part 0: Environment (nothing to submit)

Any GPU with 12 GiB or more: Delta's A40s through the course allocation
(`slurm/` has batch scripts in A4's form), the course cluster, or a cloud
VM. One GPU is enough for everything.

```bash
git clone git@github.com:csci5942/assignment-5.git
cd assignment-5
python3 -m venv .venv            # on Delta: module load pytorch-conda/2.12 first,
source .venv/bin/activate        # then add --system-site-packages to reuse its torch
pip install -r requirements.txt
python check_env.py
python data/prepare_e2e.py       # 12 MB download, writes data/e2e/
python -m pytest tests/ -q       # 1 passes; Part 2 and Part 4 tests fail with NotImplementedError
```

`test_pretrained.py` downloads GPT-2 small (about 550 MB) on first run
and checks the loaded model against a pinned loss. Then:

```bash
python sample.py
python sample.py --model gpt2-medium --prompt "The capital of France is"
```

Sizes: `gpt2` (12 layers, width 768), `gpt2-medium` (24, 1024),
`gpt2-large` (36, 1280), `gpt2-xl` (48, 1600). Every graded run uses
`gpt2-medium`, the size in the LoRA paper's Table 3.

`data/e2e/` is the E2E NLG Challenge (Novikova et al. 2017): 42,061
training pairs over 4,862 meaning representations, a dev set, and a test
set of 630 meaning representations with about seven references each.
One pair:

```
mr:  name[The Vaults], eatType[pub], priceRange[more than £30], customer rating[5 out of 5], near[Café Adriatic]
ref: The Vaults pub near Café Adriatic has a 5 star rating.  Prices start at £30.
```

The test set is written grouped by meaning representation; BLEU is
scored against all references (see `data/prepare_e2e.py`).

## Part 1: The memory model

Predict before you run. Lecture 12 counted sixteen bytes per trainable
weight: weight, gradient, two Adam moments. A4's trainer keeps fp32
weights and computes in bf16 under autocast, so here it is 4 + 4 + 8 = 16
bytes per trainable weight and 4 bytes per frozen weight. `train.py`
reports the three buckets (`params`, `grads`, `optimizer`) from the
parameter counts, the measured peak from
`torch.cuda.max_memory_allocated()`, and the difference as
`activations_residual`, as A4's `memory_report` did.

Fill in the table by hand first. Parameter counts:
`model.from_pretrained(name).num_params()`. A LoRA adapter on a Linear of
shape (out, in) at rank $r$ adds $r \cdot (\text{in} + \text{out})$
parameters; the default targets are the four Linear layers in every
block (`c_attn`, both `c_proj`, `c_fc`).

| model | trainable | params + grads + optimizer, predicted | measured peak | residual |
| :--- | ---: | ---: | ---: | ---: |
| gpt2, full | | | | |
| gpt2-medium, full | | | | |
| gpt2-medium, LoRA r = 4 | | | | |
| gpt2-medium, LoRA r = 64 | | | | |
| gpt2-xl, full | | | (predict only) | |
| gpt2-xl, LoRA r = 4 | | | (predict only) | |

Then measure the first four. Peak memory is reached within a few steps:

```bash
python train.py --config configs/gpt2-small-full.json  --steps 50 --out out/mem-small-full
python train.py --config configs/gpt2-medium-full.json --steps 50 --out out/mem-medium-full
python train.py --config configs/lora-r4.json          --steps 50 --out out/mem-lora-r4
python train.py --config configs/lora-r64.json         --steps 50 --out out/mem-lora-r64
```

Each run prints the buckets and the peak and writes them to
`out/<name>/summary.json`.

**Deliverable:** the filled table, two or three sentences on where the
residual comes from and why it changes so little between full fine-tuning
and LoRA, and what the gpt2-xl rows mean on an A40 (40 GiB) and on a
24 GiB card.

## Part 2: The template and the mask

See lecture 12, "Templates and Masking". An example becomes one token
sequence, prompt then target, and only the target span is scored.
Implement two functions in `sft.py`:

- `format_example(mr, ref)` returns `(prompt_ids, target_ids)`. The
  prompt is `PROMPT` filled with the meaning representation; the target
  is a leading space, the reference, and `<|endoftext|>`. GPT-2's BPE
  merges a leading space into the next word, so `" The"` and `"The"` are
  different tokens.
- `collate(examples)` builds `(x, y)` as A1's `train.py` did: `y[t]` is
  the token after `x[t]`. Right-pad with `<|endoftext|>` to the longest
  row and set `y` to `IGNORE` ($-100$, which `F.cross_entropy` skips by
  default, so A4's `model.forward` needs no change) on prompt and
  padding positions.

```bash
python -m pytest tests/test_mask.py -q
python -c "import sft; p,t = sft.format_example('name[Alimentum], area[city centre]', 'Alimentum is in the city centre.'); print(sft.describe(p, t))"
```

The second command prints the sequence token by token with `scored` or
`-` beside each.

**Deliverable:** that printout for one training pair, the three tests
passing, and one sentence on what scoring the prompt tokens would do.

## Part 3: Full fine-tuning

One epoch of `gpt2-medium` on the training pairs, every weight
trainable, lr $5 \times 10^{-5}$, warmup and cosine decay (A4's
schedule). About three minutes on a 4090.

```bash
python train.py --config configs/gpt2-medium-full.json
python eval.py --base gpt2-medium                  # the untouched model's row
python eval.py --run out/gpt2-medium-full
```

`eval.py` generates a description for each of the 630 test meaning
representations (greedy, stop at `<|endoftext|>`), scores corpus BLEU
against all references with sacrebleu, reports slot coverage (of the
slots with a literal string value, how many appear verbatim in the
output), and measures forgetting: mean teacher-forced loss over 200
fixed 1024-token blocks of FineWeb-Edu, A4's training corpus, in nats
per token. Results go to `results/<name>.json` with five sample outputs.

The paper's Table 3 reports 68.2 BLEU for GPT-2 M full fine-tuning after
five epochs with beam search of width 10 and the official E2E scorer.
One epoch, greedy, sacrebleu lands in the low sixties. Compare your own
rows with each other.

**Deliverable:** base and fine-tuned rows (BLEU, slot coverage, dev loss,
FineWeb loss, peak memory, seconds), two outputs for the same meaning
representation before and after, and one sentence on the FineWeb column.

## Part 4: LoRA and the rank knee

Implement the three fenced blocks in `lora.py`:

- `LoRALinear.__init__`: freeze the wrapped Linear, set `self.scale` to
  $\alpha / r$, create `A` (shape `(r, in)`, normal, std 0.01) and `B`
  (shape `(out, r)`, zeros). With `B` at zero the adapted model equals
  the base model at step 0.
- `LoRALinear.forward`: $W_0 x + \tfrac{\alpha}{r} B A x$.
- `merge_lora`: fold $\tfrac{\alpha}{r} B A$ into each wrapped weight and
  put the plain `nn.Linear` back. `eval.py` calls this before generating.

`apply_lora` is given: it freezes everything and wraps the Linear layers
named in `targets` (default: all four per block).

```bash
python -m pytest tests/test_lora.py -q
for r in 1 4 16 64; do python train.py --config configs/lora-r$r.json; done
for r in 1 4 16 64; do python eval.py --run out/lora-r$r; done
python plot_knee.py                                 # writes knee.png
```

Same steps and batch as Part 3; lr $2 \times 10^{-4}$ (the paper's
GPT-2 M setting), $\alpha = 2r$ (lecture 12). `plot_knee.py` draws dev
loss and BLEU against trainable parameters with full fine-tuning as a
horizontal line. Lecture 12 showed this curve on the A1 model; report
what yours does here. Table 3's row for this comparison: 70.4 BLEU with
0.35M trainable parameters against 68.2 with 354.9M.

**Deliverable:** `knee.png`, the five-row table (four ranks plus full),
and a paragraph: where the knee is, whether LoRA reaches full fine-tuning
on BLEU and on dev loss, and which method forgot more.

## Part 5: Which matrices

Section 7.1 of the paper: at a fixed number of trainable parameters,
which matrices to adapt. Three configs spend about 1.5M parameters:
`budget-attn-r16` (rank 16 on `c_attn` only, q, k and v in one matrix),
`budget-mlp-r12` (rank 12 on `c_fc` only), and `lora-r4` from Part 4
(rank 4 on all four).

```bash
python train.py --config configs/budget-attn-r16.json && python eval.py --run out/budget-attn-r16
python train.py --config configs/budget-mlp-r12.json  && python eval.py --run out/budget-mlp-r12
```

**Deliverable:** the three-row table (trainable, dev loss, BLEU) and two
sentences: where a fixed budget should go, and whether that agrees with
the paper's Table 5.

## The question

Half a page at most: **when is updating every weight worth it?** Cite
the Part 1 table for memory, the knee for quality, and the FineWeb column
for forgetting, and say which of the three changes your answer for a
model ten times this size.

In practice this is done with the Transformers and PEFT libraries on a
current model. The code has the same shape; only the loader changes.

## Deliverables

One PDF (Part 1 table and sentences; Part 2 printout and tests; Part 3
rows and examples; `knee.png` with its table and paragraph; Part 5
table; the question) plus `sft.py`, `lora.py`, `out/*/summary.json` and
`results/*.json`, via Gradescope.

## Layout

```
model.py             A4's model, plus from_pretrained (do not edit)
sft.py               template and loss mask (Part 2)
lora.py              LoRALinear, apply_lora, merge_lora (Part 4)
train.py             A4's single-GPU loop with collate and memory report (do not edit)
eval.py              BLEU, slot coverage, FineWeb loss (do not edit)
sample.py            greedy generation from any model or run
plot_knee.py         Part 4's figure
data/prepare_e2e.py  downloads E2E, writes data/e2e/
data/fineweb_val.bin 2.2M held-out FineWeb-Edu tokens, A4's tokenizer
configs/             the eight graded runs
tests/               pretrained, mask, lora
slurm/               A4's batch scripts, pointed at train.py and eval.py
out/, results/       written by train.py and eval.py (gitignored)
```
