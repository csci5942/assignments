# Assignment 4: Distributed Training

CSCI 5942: AI Engineering, Fall 2026.
Due date and submission: see the course page (Gradescope).

Assignment 1 gave you a transformer that fits on one GPU. This
assignment scales out training to multiple GPUs.

You will build a small, real, multi-dimensional parallel trainer, one
dimension per part. By the end you will have written data
parallelism with overlapped gradient reduction, tensor parallelism,
and pipeline paralleism with a 1F1B schedule.
These techniques are exactly how real-world training clusters
scale jobs to thousands of GPUs.

The model is assignment 1's, widened to GPT-2's published shapes.
The data is fixed and pre-tokenized for you.

## Part 0: Environment (nothing to submit)

Everything in this assignment runs on **Delta GPU**, through the course
ACCESS allocation you set up in A0. See https://docs.ncsa.illinois.edu/systems/delta/en/latest/quick_start.html for a getting started guide.

If you do not have an allocation on ACCESS, please reach out to course
staff ASAP.

There is nothing to install. NCSA ships a PyTorch module that already
has everything this assignment needs.

```bash
git clone git@github.com:csci5942/assignment-4.git
cd assignment-4
mkdir -p logs
module load pytorch-conda/2.12
conda activate base
```

### Delta GPU
You will be submitting jobs via `slurm` to the Delta GPU HPC cluster. See https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html

The above resource will have descriptions of the various hardware (partitions)
available via DeltaGPU, and their cost. Given that this is a *shared resource*,
please be aware of the total GPU hours you are using. We expect
this assignment to require under 20 GPU hours (billed per A40 GPU).
We will be monitoring usage, so do not drastically exceed the estimated
GPU time.

Develop on one `gpuA40x4` node. You need two nodes only for Part 4.

Note that our course account is `bidb-delta-gpu`.

Some tips for development:

1. **Debug on a CPU when possible.** Parts 1
   through 3 all spawn their ranks with the `gloo` backend, which needs
   no GPU at all.
2. **Debug with one GPU, not four.** `--gres=gpu:1` and a short
   `--time`.
3. **Stay on `gpuA40x4`.** Nothing in this assignment needs another
   partition. You may experiment with the A100 and H200 queues,
   but use them sparingly as they bill at a higher rate than the A40s.
4. **Use batch, not interactive jobs.** Use `sbatch` when possible
   as it releases your allocation when completed. Use `srun` only when
   needed and sparingly.

### Launching

We have provided an example batch script you can use under `slurm/`, which
runs the provided `trainer/train.py` with 1 GPU.
You may use it for example as follows. Namely, you can pass additional
arguments
to the training script.

```bash
sbatch slurm/test.sbatch --config configs/small.json
```

Observe your jobs with `squeue --me`. See the above link for other commands.

The output of each job will be placed in `logs/<job-name>-<job-id>.out`,
relative to wherever you ran `sbatch` from — so `logs/a4-test-3319470.out`
for the command above.

`trainer/train.py` writes per-step timings to `out/<name>/log.csv` and a
summary of the run
— throughput, MFU, peak memory, final loss — to `out/<name>/summary.json`.
Pass `--tag` to keep runs apart when you sweep, since `out/large/` would
otherwise be overwritten by the next `large` run.

### Start early!

`gpuA40x4` is shared with everyone who uses NCSA. During busy periods,
your job can queue for hours. Don't wait until the last minute to
start your assignment, as you may not be able to run your job in time.
Some tips:

### Reading the numbers

**Tokens per second** is the global batch divided by the median
steady-state step time.

**MFU**, model FLOPs utilization, is the fraction of the GPU's
arithmetic you actually used:

$$ \text{MFU} = \frac{\text{FLOPs per token} \times \text{tokens per second}}{\text{device peak FLOP/s}} $$

**Peak memory** is `torch.cuda.max_memory_allocated()`, reported split into
different buckets.

### The data

The training corpus is a sample of FineWeb-Edu, tokenized once with
GPT-2's BPE vocabulary (50257 tokens) and staged for you on Delta:

```
/work/nvme/bidb/csci5942/a4/fineweb/train_*.bin    ~9B tokens
/work/nvme/bidb/csci5942/a4/fineweb/val.bin        held out
```

Do not copy it into your home directory. `trainer/data.py` memory-maps
these shards and is given.

### The ladder

`configs/` holds GPT-2's four published shapes plus one model
that scales beyond the original paper.
The parameter counts run a little above the GPT-2 ones because
we do not tie the input and output embeddings.
Context length is 1024 everywhere.

| config | layers | width | heads | params | params + grads + optimizer |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `small` | 12 | 768 | 12 | 163M | 2.4 GiB |
| `medium` | 24 | 1024 | 16 | 406M | 6.1 GiB |
| `large` | 36 | 1280 | 20 | 838M | 12.5 GiB |
| `xl` | 48 | 1600 | 25 | 1.64B | 24.4 GiB |
| `huge` | 48 | 2560 | 32 | 4.04B | **60.1 GiB** |


There are also two small configs you will not report on: `smoke.json`
and `verify.json`, which may be helpful for testing correctness.

Confirm the whole thing runs before you change anything:

```bash
python -m pytest tests/ -q
sbatch slurm/test.sbatch        # one GPU, 30 steps of `small`
```

Note that tests will fail due to `NotImpelmentedError`s from code
you have not written yet.

### Ranks and groups
Some terminology that comes up in this assignment:

A **rank** is one process's global index in the job, `0` to
`world_size-1`, one process per GPU.

`local_rank` is the one you hand to `torch.cuda.set_device`.
This is the **node-local** rank, which is unique per node,
while the **global** rank is unique across nodes.

| | what it is | where it comes from |
| :--- | :--- | :--- |
| `rank` | index across the whole job | `RANK`, or Slurm's `SLURM_PROCID` |
| `world_size` | total processes | `WORLD_SIZE` / `SLURM_NTASKS` |
| `local_rank` | index of this process's GPU **on its own node** | `LOCAL_RANK` / `SLURM_LOCALID` |

A **process group** is a subset of ranks that run collectives among
themselves. The default group is everyone; an `all_reduce` over a
smaller group touches only its members, and the ranks outside it are
neither involved nor blocked.
With three parallelism dimensions you need different collectives
over different subsets at the same time.

This is set up for you automatically by `trainer/mesh.py`.
Specifically:

```
rank = pp_rank·(dp·tp) + dp_rank·tp + tp_rank
```

For example, with `(dp=2, tp=2, pp=2)` on eight GPUs, rank 3 is
`(pp=0, dp=1, tp=1)`,
and its three groups are the ranks matching it on every *other* axis:

| group | rank 3's partners | what crosses it |
| :--- | :--- | :--- |
| tp | 2 | activations, twice per block |
| dp | 1 | gradients, once per step |
| pp | 7 | activations, at stage boundaries |

Run `python -m trainer.mesh --dp 2 --tp 2 --pp 2` to print the whole
grid.

### Communication cost

`bench/allreduce.py` is given. It all-reduces
buffers from 1 MB to 1 GB and reports **bus bandwidth**: the bytes
actually crossing the wire per second, which for a ring all-reduce over
`P` ranks is

$$ \text{busbw} = \frac{2(P-1)}{P} \cdot \frac{N}{t} $$

for an `N`-byte buffer taking `t` seconds.

```bash
sbatch slurm/bench_a40.sbatch        # about three minutes, 0.2 GPU-hours
```

It reports two curves, at two and at four A40s.
**Keep them somewhere you can find them**, as later parts use this number.

## Part 1: Data parallel

Implement the missing functions in `trainer/ddp.py`, which
implement data parallel training.

### Get it right first

`broadcast_module_state` makes every rank's copy of the model identical
before the first step.

`naive_all_reduce_grads`: after `loss.backward()`, all-reduce
every gradient across the data-parallel group.

Test via the following.

```bash
python -m pytest tests/test_ddp.py -q
```

### Then make it fast

The naive version waits for the whole backward pass to finish and then
sits idle during one large all-reduce. However gradients
become available *as the backward pass walks the layers*, so the
reduction for the last layer can be in flight while the first layer is
still computing.

Implement `OverlappedDDP`: register a post-accumulate-grad hook on every
parameter and launch an asynchronous all-reduce as soon as a gradient
lands, then wait on all of them before the optimizer step.

Measure `medium` on 1, 2, and 4 GPUs, on the A40 partition at a fixed
global batch. The script asks for 4 GPUs by default; override the
allocation on the `sbatch` line for the other two points:

```bash
sbatch slurm/scale_a40.sbatch --config configs/medium.json
sbatch --gpus-per-node=2 --ntasks-per-node=2 \
       slurm/scale_a40.sbatch --config configs/medium.json
sbatch --gpus-per-node=1 --ntasks-per-node=1 \
       slurm/scale_a40.sbatch --config configs/medium.json
```

**Deliverable.** A table of tokens/second and scaling efficiency for
naive and overlapped, at 1, 2 and 4 A40s. Then:

1. Compare your measured naive 4-GPU iteration time to the ideal
   scaling time with 4x scaling. Much of the overhead is due to
   communication. Calculate how many bytes a ring all-reduce moves
   per step, divide that by your measured
   part 0 bus bandwidth, and compare it to the overhead you measured at
   4 GPUs. How close is it?
2. Your overlapped version hides some of the overhead. How much of the
   predicted communication time does it actually hide, and what sets
   the ceiling on how much it could?

## Part 2: Tensor parallel

Implement the six stubs in `trainer/tp.py`:

- `_F.backward` and `_FBar.forward`. `f` is an identity forward and an
  all-reduce backward; `f_bar` is the other way around.
- `ColumnParallelLinear.forward` and `RowParallelLinear.forward`.
  Column-parallel splits `W` by output column and needs the full input;
  row-parallel splits by input row and produces a partial sum. Stacked,
  column-parallel's output is already sharded the way row-parallel wants
  its input, so the pair needs one collective, not two.
- `TensorParallelMLP.__init__` and `.forward` to perform that stack:
  `c_fc` column-parallel, `c_proj` row-parallel, followed by one
  all-reduce.

```bash
python -m pytest tests/test_tp.py -q
```

### Run TP

Train `large` at the same global batch in four configurations: a
single-GPU baseline, then DP=4, TP=4 and DP=2 x TP=2, each across all
four GPUs.

```bash
sbatch slurm/tp_a40.sbatch --config configs/large.json
```

**Deliverable.** Tokens/second and peak memory for all four
configurations, as one table. Then answer:

1. Rank the four by throughput. TP=4 and DP=4 do the *same* arithmetic
   on the same four GPUs, so explain the throughput gap between them. 
   What is the benefit of TP over DP? 
   Support your answer with measurements. Would
   you use TP or DP for this model and this GPU configuration?
2. Work out how many bytes DP=4 and TP=4 each put on the wire in one
   step. Which moves more, and by how much? Divide each by your
   measured bus bandwidth -- does that account for the throughput
   gap between TP=4 and DP=4?

## Part 3: Pipeline parallel

Most of pipeline parallelism is given for you, namely an All Forward
All Backward implementation. Understand the implementation.
Your job is to then implement 1F1B in `trainer/pp.py`.

- `one_forward_one_backward`. Once the pipe is full each stage
  alternates one forward with one backward. The bubble is identical to
  AFAB's, but at most `p` micro-batches are ever in flight, so
  activation memory stops growing with `m`.

```bash
python -m pytest tests/test_pp.py -q
```

### Measure

One `gpuA40x4` node, `large` at `pp=4`:

```bash
sbatch slurm/pp_a40.sbatch
```

**Deliverable.** Run your pipeline for at least three values of `m`.
Report the measured idle fraction (`train.py` reports a
`busy_fraction`) and peak memory, under AFAB and under 1F1B. Then
answer:

1. How does your measured bubble change with `m`?
2. What does that change in `m` cost you in memory, under each
   schedule?

## Part 4: Hybrid parallelism

Everything so far has run on a model that fits on one GPU.
`configs/huge.json` is 4.04B parameters, which needs **60.1 GiB** for
weights, gradients and Adam's two moments before a single activation.
Try to train it and observe what happens.

```bash
sbatch slurm/part4_a40.sbatch --steps 6 --warmup-steps 1 \
       --dp 8 --tp 1 --pp 1 --ddp naive --tag oom
```

Your job is to find the mapping that trains `huge` fastest, using 8 GPUs
across 2 nodes (2 groups of 4).

Keep the model, the data and its order, the seed, the global batch of
128, the optimizer, and two `gpuA40x4` nodes fixed. We also
suggest keeping `tp < 8`.

Run it as follows:

```bash
sbatch slurm/part4_a40.sbatch --steps 6 --warmup-steps 1 \
       --dp <DP> --tp <TP> --pp <PP> --tag <tag> --schedule <pipeline_schedule_config> --ddp <ddp_config>
```

**Deliverable.** Include a table of every configuration you tried, with
step time, tokens/second, peak memory and loss where available. Mark runs
that failed to fit. Identify your best configuration, then explain why it
wins: what each degree of sharding bought in memory, what it cost in time,
and why your chosen configuration was the best.

### (+.25) Bonus: continue scaling your model.

Continue scaling your model. How large of a model can you fit in 8 GPUs?
Write your own config and find out. Start from `configs/huge.json` and
continue scaling the model configuration.

**Deliverable.** The largest model you trained: its config, its mapping,
its step time and its peak memory. Explain your final configuration and
what prevented you from scaling further.

## Deliverables

Submit your code plus one PDF.

**Code**

- `trainer/ddp.py`, `trainer/tp.py`, `trainer/pp.py`
- The configuration you ran in Part 4 (and the bonus configuration).

**Report**

Include your deliverables to parts 1-4.
