# Assignment 1: Transformers on Shakespeare, from scratch

CSCI 5942: AI Engineering, Fall 2026. Released Thu Aug 27.
Due date and submission: see the course page (Gradescope).

You will implement a decoder-only transformer with no libraries beyond
PyTorch, train it on the complete works of Shakespeare at the character
level, and run a small scaling study on your trained models. Later
assignments (evaluation, SFT and PEFT, RLAIF) build on this model, so
keep your trained checkpoints.

AI usage follows the course policy: permitted as if it were a
colleague, and you are responsible for your work. For this assignment
in particular, the model file is short enough to understand completely;
if you cannot rewrite `model.py` from memory afterwards, the assignment
has not done its job, whoever wrote your first draft.

## Part 0: Environment (nothing to submit)

Python 3.11+, PyTorch 2.x. Google Cloud credits come from the signup in
week 1.

```bash
pip install torch numpy pytest
python data/prepare.py            # downloads ~1.1MB, builds train/val bins
python train.py --config configs/smoke.json   # ~1 min sanity run
```

The smoke run must reach val loss below 3.0. If it does not, your
environment (not your model) is the problem to fix first.

### What needs a GPU, and what does not

**Parts 1 and 2 do not.** The four tests finish in about two seconds on a
laptop, and `configs/smoke.json` is a one-minute CPU run. You can write
and debug the entire model on a CPU, so start before your cloud credits
arrive rather than after.

**Part 3 does.** Wall-clock for all 5,000 iterations of each rung:

| | pico | nano | micro | mini | base |
| :--- | ---: | ---: | ---: | ---: | ---: |
| one RTX 4090 | 0.4 min | 0.4 min | 0.5 min | 1.4 min | 1.9 min |
| 32-core CPU | ~35 min | ~70 min | hours | hours | hours |

The whole ladder is under five minutes on one consumer GPU and runs to
about two days on a CPU. A cloud T4 sits between the two, closer to the
GPU column. Run Part 3 on your Google Cloud instance.

## Part 1: Implement the model

`model.py` has the module skeletons with shapes documented; every
`NotImplementedError` is yours. Implement, in order:

1. `CausalSelfAttention.__init__` / `forward`: fused qkv projection,
   head reshape, scaled dot-product scores, causal mask, softmax,
   value mix, output projection.
2. `MLP`: expand 4x, GELU, project back.
3. `Block`: pre-norm residual wiring.
4. `GPT.forward` body: token + position embeddings, the block stack,
   final norm, logits.

Check yourself as you go:

```bash
python -m pytest tests/ -q
```

The causality test catches the two classic bugs (mask applied after
softmax, mask oriented the wrong way). Do not train until all four
tests pass.

## Part 2: Train it

```bash
python train.py --config configs/base.json
python sample.py --run out/base --prompt "DUKE OF" --tokens 400
```

The base config (6 layers, 384 wide, ~10.6M non-embedding params)
reaches val loss near 1.47 in a few minutes on a modern GPU. Include
in your report: the train/val curves from `out/base/log.csv`, a 400
character sample, and two sentences on where the train and val curves
separate and why.

## Part 3: A scaling ladder

Train the remaining three sizes (each config states its size; all four
see the same 82M training tokens):

```bash
for c in nano micro mini; do python train.py --config configs/$c.json; done
```

Then fit the saturating power law

$$L(N) = c + a N^{-\alpha}$$

to your four (N, best val loss) points, using any fitting method you
can defend. Report your fitted alpha and c, plot the four points with
the fitted curve on log-log axes, and answer:

1. Chinchilla trained at roughly 20 tokens per parameter. Where does
   your base run sit on that scale, and what does that predict about
   which side of the compute-optimal frontier these runs are on?
2. Your fitted c is not zero. What does a nonzero floor mean for
   character-level Shakespeare specifically?
3. If you could double one thing (parameters or training tokens),
   which does your fit say to double, and how confident are you?

## Deliverables

One PDF (curves, sample, fits, answers) plus your `model.py`, via
Gradescope. Expected effort breakdown: Part 1 is the majority of the
learning; Part 3 is the majority of the writing.

## Layout

```
data/prepare.py   char-level tokenization, train/val split
model.py          the transformer (your implementation)
train.py          training loop, CSV logging, checkpointing
sample.py         generation from a checkpoint
configs/          smoke + the four ladder sizes
tests/            shape, causality, init, and overfit checks
```
