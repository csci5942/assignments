# Assignment 6: Serving GPT-2

CSCI 5942: AI Engineering, Fall 2026. Released Tue Oct 20, due Tue Nov 3
at 11:59 PM on Gradescope.

This assignment turns your model implementation that you've been working
on into a full-fledged inference serving engine.
In part 1, you will implement KV caching; part 2 will implement PagedAttention;
and part 3 will add continuous batching to your serving engine.

## Files

| file | overview |
| :--- | :--- |
| `model.py` | Your A1 transformer, plus `from_pretrained()` (from A5) to load GPT-2's weights. **Part 1:** `KVCache`. The cached `generate` is given. |
| `paged.py` | **Part 2:** `BlockAllocator` and `PagedKVCache`. A batched `generate` over the paged cache is given. |
| `engine.py` | `Request` and the serving loop (given). **Part 3:** `Scheduler`. |
| `sample.py` | Generates from GPT-2 with and without the cache and checks that they agree. |
| `bench_kv.py` | KV cache benchmark script for measuring decode time and KV memory used. |
| `bench_serve.py` | Serving benchmark script for measuring throughput and latency. |
| `check_env.py` | Environment setup script. |
| `tests/` | `test_model.py` (A1's), `test_kvcache.py`, `test_paged.py`, `test_engine.py`. |

## Part 0: Environment (nothing to submit)

Like assignment 5, you can use any GPU with bf16 support (Ampere or
newer: A40, A100, L4, ...) and ideally at least 16 GB of memory.
`check_env.py` checks both.
You may use Delta, the CURC cluster, Chameleon, a cloud VM, or another
GPU you can access. Please clarify the compute node that you used in your report.

```bash
cd assignment-6
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python check_env.py
python -m pytest tests/ -q       # A1's 4 pass; the rest fail with NotImplementedError
```

The first `from_pretrained` downloads OpenAI's weights from the Hugging
Face Hub (no account needed). gpt2-xl is 6.4 GB.

## Part 1: The KV cache

Your first task is to implement KV Caching on top of the model
you've been building.

Without a cache, `generate` runs the whole sequence through the model for
every new token. With one, the model runs on the new tokens alone: each
layer stores their keys and values, and attention reads every earlier
token's keys and values from the cache instead of recomputing them.

The model and the cache talk through three methods, which `GPT.forward`
calls on every forward pass, in this order:

| call | when | what it is for |
| :--- | :--- | :--- |
| `begin(T)` | once, before the embeddings | Returns the position ids of the T new tokens. GPT-2 adds a learned embedding for each token's absolute position (`wpe`). Without a cache, `idx` is the whole sequence and the positions are `0 .. T-1`. With one, `idx` holds only the new tokens, so their positions need to start at the number of tokens already cached. |
| `update(layer, k, v)` | in every layer | Stores the new tokens' keys and values, and returns every key and value cached so far, with a mask saying which of them each new token may attend to. |
| `end(T)` | once, after the last layer | Records that T more tokens are cached (given). |

Implement, in `model.py`:

1. `KVCache.__init__`: Allocate K and V tensors.
2. `KVCache.begin`: Calculate the position IDs.
3. `KVCache.update`: write the new keys and values at slots
   `pos .. pos+T-1`, and return every filled slot and the mask.

`GPT.generate`'s cached path is given: one forward over the prompt
(prefill), then one per new token (decode).

Batches of prompts with different lengths are padded on the left
(`left_pad`), so every row's next token goes into the same slot.

Once complete, run the following.

```bash
python -m pytest tests/ -q
python sample.py --model gpt2-xl --tokens 200
python sample.py --model gpt2-xl --tokens 800
python sample.py --prompt "The capital of France is" "Once upon a time" --tokens 50
```

The tests compare every cached logit with A1's uncached forward, on a
small random model. `sample.py` checks that both paths produce the same
greedy tokens and reports milliseconds per token.

**Deliverable:** your completed `model.py`, with the tests passing. In
the report:
- ms/token for gpt2-xl, cached and uncached, at 200 and 800 new tokens
  (the two `sample.py` runs above);
- two or three sentences on why the uncached time grows with the number
  of tokens and the cached time does not;
- the size of the KV cache for one 1024-token gpt2-xl request in bf16,
  showing your working.

## Part 2: PagedAttention

In Part 2, you will implement PagedAttention. A contiguous cache reserves
`max_len` slots for every sequence, whether or not it uses them. A paged
cache keeps keys and values in one pool of fixed-size blocks, allocated
once, and gives a sequence a block only when it needs one. Each sequence
has a block table: the ids of its blocks, in order, mapping its logical
blocks to physical ones. A `BlockAllocator` hands out free block ids;
`PagedKVCache` owns the pool and the block tables, and gives a
sequence's blocks back when it finishes.

Sequences in a batch each have their own length, so the pool never
reserves space for padding. The per-step tensors are still padded to the
longest sequence, because tensor operations need rectangular shapes.
Short rows of the block table are padded with block id 0. Block 0 is an
ordinary block that some other sequence may be using, but the mask in
`update` hides those padded slots, so whatever it holds is never read
(see `begin` and `update`).

`PagedKVCache` has the same `begin`, `update` and `end` as `KVCache`, so
`model.py` does not change. Implement, in `paged.py` (each method's
docstring has the details, and the `PagedKVCache` docstring works
through an example):

1. `BlockAllocator`: a free list of block ids. `allocate()` takes one
   (and raises `MemoryError` when none is left), `free(blocks)` puts them
   back, `num_free` counts them.
2. `PagedKVCache.begin`, once per forward pass:
   - give each sequence in the batch enough blocks for `lens[s] + T`
     tokens;
   - set `positions` (the new tokens' positions), `block_table` (each
     row's block ids, padded with 0) and `slots` (each new token's flat
     slot in the pool: for a token at position `p` in sequence `s`,
     `tables[s][p // block_size] * block_size + p % block_size`).
3. `PagedKVCache.update`, in every layer, three short blocks:
   - write: reshape the new keys and values to one row per token, and
     write them to their slots in the pool;
   - gather: copy each sequence's blocks out of the pool, in table
     order, into one tensor of shape `(B, n_head, S, head_dim)`, where
     `S = max_blocks * block_size`;
   - mask: new token `t` of row `b` may attend to slot `s` when
     `s <= positions[b, t]`.
4. `PagedKVCache.remove`: give a finished sequence's blocks back to the
   allocator, and delete its entries.

The gather copies every sequence's keys and values in every layer.
Real LLM serving systems use a special attention kernel that can
read non-contiguous blocks straight from the pool.
To avoid having you implement that functionality, we add an additional
gather operation to *actually* make the blocks contiguous.
However, this adds significant overhead due to an additional memory copy.

```bash
python -m pytest tests/ -q
python bench_kv.py --model gpt2-xl --batch 32 --tokens 128 --block-size 8 16 32 64 128 256
```

`bench_kv.py` runs 32 prompts of 16 to 512 tokens through both caches
and reports decode milliseconds per step and the fraction of reserved
KV slots that hold a token.

**Deliverable:** your completed `paged.py`, with the tests passing. In
the report:
- the `bench_kv.py` table (decode ms/step and the fraction of reserved
  slots used, for each cache and block size);
- how many of these requests fit in 4 GB of KV cache, contiguous
  (reserving 1024 slots each) versus paged at block size 16 , showing your working;
- the block size you would use, with one sentence on what very small
  and very large blocks each cost.

## Part 3: Continuous batching

In Part 3, you will write the scheduler of a small serving engine built
on your paged cache. A static batcher runs a batch until its longest
request finishes, so the slots of requests that finish early sit idle. A
continuous batcher retires finished requests and admits new ones at every
iteration, so freed blocks are reused right away.

`engine.py` gives the serving loop. Each iteration it calls your
scheduler, prefills the newly admitted requests one at a time, then runs
one decode step for every running request. Implement, in `engine.py`:

1. `Scheduler.can_admit`: whether the pool has room for a request next
   to the running ones. Ensure that each running request
   has enough blocks to execute, and admit only if the 
   new request fits in what is left. What a request needs depends on the
   admission rule: its full length (`prompt + max_new`) by default, or,
   with `optimistic=True`, only the tokens so far plus the next one.
2. `Scheduler.schedule`, at the start of every iteration:
   - retire finished requests and give their blocks back
   - admit waiting requests in first-come, first-served order while a
     batch slot is free and `can_admit` allows. 

`Scheduler.preempt` is given. With optimistic admission the pool can run
out in the middle of a decode step; the engine then calls `preempt`,
which evicts the most recently admitted request and puts it back at the
front of the queue, to be recomputed when it is readmitted (vLLM's
default).

```bash
python -m pytest tests/ -q
python bench_serve.py --kv-gb 4
python bench_serve.py --kv-gb 1
python bench_serve.py --kv-gb 1 --engine continuous --optimistic
```

`bench_serve.py` serves 400 requests, all arriving at once, with prompts
of 16 to 64 tokens and outputs of 8 to 128 (EOS ignored).

**Deliverable:** your completed `engine.py`, with the tests passing. In
the report:
- the `bench_serve.py` tables: static and continuous at 4 GB and at
  1 GB, and continuous with optimistic admission at 1 GB.
- static against continuous, at both budgets: compare their tokens/s,
  mean latency and mean batch size. Does continuous improve each of
  these three metrics, and if so, why? One sentence per metric.
- what paging bought, at 1 GB of KV cache. A contiguous cache must
  reserve 1024 slots for every request, so:
  1. how many requests fit in 1 GB at once? (Use your Part 1 KV size
     per 1024-token request.)
  2. roughly what tokens/s would running the maximum concurrent
     requests for contiguous cache reach, based on your observations
     from previous parts?
  3. compare with your continuous run's tokens/s at 1 GB, and discuss
     why paging improves/worsens token throughput in a sentence or two.
- compare optimistic admission against safe admission at 1 GB. What is
  the change in tokens/s and mean batch? What are the benefits and costs
  of optimistic admission?

## Deliverables

One PDF plus your `model.py`, `paged.py` and `engine.py`, via Gradescope.
