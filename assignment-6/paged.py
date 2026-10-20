"""PagedAttention (Part 2).

A contiguous cache reserves `max_len` slots for every sequence, whether
or not it uses them. A paged cache, as in vLLM, keeps keys and values in
one pool of fixed-size blocks and gives a sequence a block only when it
needs one. Four things for you to implement in this file.

    BlockAllocator         the free list of block ids
    PagedKVCache.begin     assign blocks, and work out where new tokens go
    PagedKVCache.update    store new keys and values, gather each sequence
    PagedKVCache.remove    give a finished sequence's blocks back

`PagedKVCache` plugs into `GPT.forward` exactly like Part 1's `KVCache`,
through `begin`, `update` and `end`, so `model.py` does not change.

To run the correctness suite

    python -m pytest tests/test_paged.py -q
"""

import torch

from model import GPT, GPTConfig, sample


class BlockAllocator:
    """A free list over the pool's block ids, `0 .. num_blocks-1`."""

    def __init__(self, num_blocks: int):
        """Build a data structure to track free block ids, with a total
        number of blocks `num_blocks`.
        Start with every block id free."""
        self.num_blocks = num_blocks
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    def allocate(self) -> int:
        """Take a free block id and return it.

        Raise `MemoryError` if none is left. Each id is handed out only once 
        until it is freed.

        Returns the allocated block id.
        """
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    def free(self, blocks: list[int]):
        """Put `blocks` (block ids) back on the free list."""
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    @property
    def num_free(self) -> int:
        """Returns how many block ids are free."""
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")


class PagedKVCache:
    """Keys and values for every layer, in one pool of blocks.

    The pool is allocated once, in `__init__`: `k[i]` and `v[i]` have shape
    `(num_blocks, block_size, n_head, head_dim)` for layer `i`. A block is
    named by its **id**, its index in the pool, so block 3 of layer `i`'s
    keys is `k[i][3]`.

    Each sequence `seq` has

    * `tables[seq]`, its block table: the ids of its blocks, in order
    * `lens[seq]`, the number of tokens it has cached.

    Token `j` of `seq` lives in block `tables[seq][j // block_size]`, at
    offset `j % block_size`.

    **Flat slots.** `update` writes into the pool with one number per
    token, which points to the "physical address" in the entire pool
    of the new token: `block_id * block_size + offset`. For example, with
    `block_size = 4`:

        block id     0           1           2         ...  5
        offset       0  1  2  3  0  1  2  3  0  1  2  3     0  1  2  3
        flat slot    0  1  2  3  4  5  6  7  8  9 10 11    20 21 22 23

    **An example**, with `block_size = 4`. Sequence 0 has 8 tokens, in
    blocks `[3, 7]`; sequence 1 has 2, in block `[5]`. One decode step
    calls `select([0, 1])`, then `begin(1)`:

    * Sequence 0 needs room for 9 tokens and has 8, so it gets one more
      block, say 1: `tables[0] = [3, 7, 1]`. Sequence 1 needs 3 and has 4.
    * `begin` sets

          positions   = [[8], [2]]
          block_table = [[3, 7, 1], [5, 0, 0]]      row 1 padded with 0
          slots       = [[1*4 + 0], [5*4 + 2]] = [[4], [22]]

    * In every layer, `update` writes the new keys and values to slots 4
      and 22, gathers blocks `[3, 7, 1]` and `[5, 0, 0]`, and masks row 0
      to slots 0-8 and row 1 to slots 0-2.
    """

    def __init__(self, config: GPTConfig, num_blocks: int, block_size: int,
                 device=None, dtype=None):
        self.block_size = block_size
        self.max_len = config.block_size
        self.allocator = BlockAllocator(num_blocks)
        self.tables: dict[int, list[int]] = {}
        self.lens: dict[int, int] = {}
        self.batch: list[int] = []
        # Set by begin() once per step, read by every layer's update():
        #   positions (B, T): the new tokens' positions
        #   block_table (B, max_blocks): each row's block ids, padded
        #   slots (B, T): each new token's flat slot in the pool
        self.positions: torch.Tensor | None = None
        self.block_table: torch.Tensor | None = None
        self.slots: torch.Tensor | None = None
        hd = config.n_embd // config.n_head
        shape = (num_blocks, block_size, config.n_head, hd)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]

    def add(self, seq: int):
        """Start an empty sequence with id `seq`."""
        self.tables[seq], self.lens[seq] = [], 0

    def remove(self, seq: int):
        """Give `seq`'s blocks back to the allocator, and forget `seq`.

        Called when a sequence finishes (or, in Part 3, is preempted). Delete
        its entries in `self.tables` and `self.lens`.
        """
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    def select(self, seqs: list[int]):
        """Choose the sequences in the next forward pass, by id."""
        self.batch = list(seqs)

    def begin(self, T: int) -> torch.Tensor:
        """Assign blocks, and work out where the `T` new tokens go.

        Called once per forward pass, before the embeddings, for every sequence
        in `self.batch`.

        What to write:

        * Give each sequence `s` enough blocks for `lens[s] + T` tokens, by
          appending ids from `self.allocator` to `tables[s]`. Raise
          `ValueError` if past `max_len` and `MemoryError`
          if the pool runs out of memory.

        * Set the three tensors declared in `__init__`. `self.batch` holds
          the ids of the sequences in this pass (see `select`): row `b` is
          the sequence whose id is `s = self.batch[b]`.

          - `self.positions` `(B, T)`: `lens[s] .. lens[s]+T-1`.
          - `self.block_table` `(B, max_blocks)`: each row's block ids,
            `tables[s]`, padded on the right with 0 to the longest row. These
            are ids, not the blocks' contents.
          - `self.slots` `(B, T)`: each new token's flat slot. For a
            token at position `p`, that is
            `tables[s][p // block_size] * block_size + p % block_size`.

        Return `self.positions`. Leave `lens` alone; `end` updates it. The
        class docstring works through an example.
        """
        dev = self.k[0].device
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Store the new keys and values, and gather each sequence's blocks.

        Called in every layer with `k` and `v` of shape
        `(B, n_head, T, head_dim)`. Returns `(K, V, mask)`: `K` and `V` of
        shape `(B, n_head, S, head_dim)` with `S = max_blocks * block_size`,
        and a bool `mask` of shape `(B, 1, T, S)`.

        What to write, in the three marked blocks below:

        * Write: put each new token's key and value into its flat slot.

        * Gather: index the pool with `self.block_table` to fetch each row's
          blocks, in table order. Reshape the result so slot `s` of a row is
          that sequence's token at position `s`. Same for `V`.

        * The mask: new token `t` of row `b` may attend to slot `s` when
          `s <= positions[b, t]`. That also hides the padding. It is the
          same comparison as Part 1's causal mask.

        **The gather stands in for a special kernel.** It copies every
        sequence's keys and values, in every layer. Real systems avoid the copy
        with an attention kernel that reads blocks straight from the pool.
        Since PyTorch doesn't have a specialized kernel, we approximate
        it with an (expensive) copy here.
        """
        B, nh, T, hd = k.shape
        S = self.block_table.size(1) * self.block_size

        # 1. Write. flat_k is this layer's key pool viewed as a list of
        #    slots: shape (num_blocks * block_size, nh, hd), where row i is
        #    flat slot i (block i // block_size, offset i % block_size). It
        #    is a view, so writing into it writes into the pool. Same for
        #    flat_v.
        flat_k = self.k[layer].view(-1, nh, hd)
        flat_v = self.v[layer].view(-1, nh, hd)
        # Reshape k and v into the appropriate shape for writing
        # into the flat pool. Then, write each new token's 
        # key and value into the right row of the flattened view of the 
        # pool, using the slot indices in self.slots.
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

        # 2. Gather. Create a contiguous tensor from self.k[layer] that
        #    groups together, for each row, all the blocks in that
        #    sequence's block table, in table order (hint: use
        #    self.block_table to index the pool). Then turn the result into
        #    the shape attention needs, (B, nh, S, hd). Same for
        #    V, from self.v[layer].
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")

        # 3. Mask, (B, 1, T, S): True where new token t of row b may attend
        #    to slot s, i.e. s <= self.positions[b, t].
        # --- YOUR IMPLEMENTATION HERE ---
        raise NotImplementedError("implement this block")
        return K, V, mask

    def end(self, T: int):
        for s in self.batch:
            self.lens[s] += T

    def stats(self) -> dict:
        """Token slots reserved vs holding a real token."""
        used_blocks = self.allocator.num_blocks - self.allocator.num_free
        return {"reserved": used_blocks * self.block_size, "used": sum(self.lens.values())}


@torch.no_grad()
def generate(model: GPT, prompts: list[list[int]], max_new_tokens: int,
             cache: PagedKVCache, temperature: float = 0.0, top_k: int | None = None):
    """Prefill each prompt on its own, then decode them as one batch.
    Returns the new tokens for each prompt and the cache's peak stats."""
    dev = cache.k[0].device
    seqs = list(range(len(prompts)))
    last = []
    for s, p in zip(seqs, prompts):
        cache.add(s)
        cache.select([s])
        logits, _ = model(torch.tensor([p], device=dev), cache=cache)
        last.append(logits[:, -1])
    logits = torch.cat(last)                                    # (B, V)
    cache.select(seqs)
    out = []
    for i in range(max_new_tokens):
        nxt = sample(logits, temperature, top_k)                # (B, 1)
        out.append(nxt)
        if i < max_new_tokens - 1:
            logits, _ = model(nxt, cache=cache)
            logits = logits[:, -1]
    peak = cache.stats()
    for s in seqs:
        cache.remove(s)
    return torch.cat(out, 1).tolist(), peak
