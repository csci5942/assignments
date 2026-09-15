"""Loader tests for Part 2.

    python -m pytest tests/test_loader.py -q

`test_shards_cover_everything_exactly_once` is the one that matters. The
classic sharding bug does not crash: every worker yields the same
shards, so you train on 1/N of your data at N times the apparent
throughput and the only symptom is a loss curve that is worse than it
should be for reasons you cannot see.
"""

import os

import numpy as np
import pytest

TEXTS = [
    "The quick brown fox jumps over the lazy dog and sleeps in the sun.",
    "It is a truth universally acknowledged that a single man in want.",
    "Call me Ishmael and some years ago never mind how long precisely.",
    "In the beginning God created the heaven and the earth and the void.",
]


@pytest.fixture
def mix(tmp_path):
    """A mix directory of Parquet shards, shaped like write_mix's output."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = tmp_path / "mix_test"
    d.mkdir()
    for i in range(6):
        rows = [f"shard {i} line {j}. {TEXTS[j % len(TEXTS)]}" for j in range(40)]
        pq.write_table(
            pa.table({"doc_id": [f"{i}-{j}" for j in range(40)],
                      "host": ["h.com"] * 40,
                      "text": rows,
                      "n_tokens": [len(r) for r in rows]}),
            d / f"part-{i:05d}.parquet")
    return str(d)


# =====================================================================
# shard_paths
# =====================================================================
def test_shard_paths_finds_all_shards(loader, mix):
    assert len(loader.shard_paths(mix)) == 6


def test_shard_paths_is_sorted(loader, mix):
    paths = loader.shard_paths(mix)
    assert paths == sorted(paths), (
        "the order fixes which worker gets which shard; an unsorted glob "
        "can differ between machines and between runs")


def test_shard_paths_empty_dir(loader, tmp_path):
    assert loader.shard_paths(str(tmp_path)) == []


# =====================================================================
# assign_shards -- the important one
# =====================================================================
class TestAssignShards:
    PATHS = [f"s{i}.parquet" for i in range(12)]

    def test_single_reader_gets_everything(self, loader):
        got = loader.assign_shards(self.PATHS, 0, 1, 0, 1)
        assert got == self.PATHS

    @pytest.mark.parametrize("world,workers", [(1, 2), (1, 4), (2, 1), (2, 2), (3, 4)])
    def test_shards_cover_everything_exactly_once(self, loader, world, workers):
        """No overlap, and nothing missed.

        This is the test that catches the classic bug. Ignoring either
        dimension -- rank or worker -- makes several readers return the
        same shards. Nothing errors; you just silently train on a
        fraction of your data.
        """
        seen = []
        for rank in range(world):
            for worker in range(workers):
                seen.extend(loader.assign_shards(
                    self.PATHS, rank, world, worker, workers))
        assert sorted(seen) == sorted(self.PATHS), (
            f"with world={world}, workers={workers} the union of all "
            "readers must be exactly the shard list")
        assert len(seen) == len(set(seen)), "a shard went to two readers"

    def test_readers_are_balanced(self, loader):
        sizes = [len(loader.assign_shards(self.PATHS, 0, 1, w, 4))
                 for w in range(4)]
        assert max(sizes) - min(sizes) <= 1, f"uneven split: {sizes}"

    def test_more_readers_than_shards(self, loader):
        """8 readers, 3 shards: five readers get nothing, and that is fine
        as long as no shard is duplicated or lost."""
        paths = ["a.parquet", "b.parquet", "c.parquet"]
        seen = [p for w in range(8)
                for p in loader.assign_shards(paths, 0, 1, w, 8)]
        assert sorted(seen) == sorted(paths)


# =====================================================================
# encode
# =====================================================================
def test_encode_roundtrips(loader):
    stoi, itos = loader.load_vocab()
    ids = loader.encode("Hello world", stoi)
    assert "".join(itos[i] for i in ids) == "Hello world"


def test_encode_drops_out_of_vocabulary(loader):
    stoi, _ = loader.load_vocab()
    assert loader.encode("café", stoi) == loader.encode("caf", stoi)


# =====================================================================
# pack
# =====================================================================
class TestPack:
    def test_block_length_is_block_size_plus_one(self, loader):
        blocks = list(loader.pack([list(range(10))] * 20, block_size=16))
        assert blocks, "no blocks emitted"
        assert all(len(b) == 17 for b in blocks), (
            "emit block_size + 1 tokens: the loop uses the first "
            "block_size as input and the last block_size as target")

    def test_consumes_across_line_boundaries(self, loader):
        """Lines are shorter than a block, so a block must span several."""
        blocks = list(loader.pack([[1, 2, 3]] * 30, block_size=8))
        assert len(blocks) >= 3

    def test_no_blocks_when_starved(self, loader):
        assert list(loader.pack([[1, 2]], block_size=64)) == []

    def test_target_is_input_shifted_by_one(self, loader):
        block = next(iter(loader.pack([list(range(100))], block_size=8)))
        x, y = block[:-1], block[1:]
        assert x[1:] == y[:-1]


# =====================================================================
# MixDataset
# =====================================================================
class TestMixDataset:
    def test_yields_correct_shapes(self, loader, mix):
        import torch
        ds = loader.MixDataset(mix, block_size=32)
        x, y = next(iter(ds))
        assert x.shape == (32,) and y.shape == (32,)
        assert x.dtype == torch.int64
        assert torch.equal(x[1:], y[:-1])

    def test_missing_mix_raises(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.MixDataset(str(tmp_path), block_size=32)

    def test_workers_do_not_duplicate_data(self, loader, mix):
        """Sharding must PARTITION the shards, not hand all of them out.

        Comparing the *sets* each worker configuration sees cannot catch
        this: if every worker reads every shard, the union is still the
        whole mix. Counting can. `pack` is called once per shard, so the
        block count of a shard does not depend on who reads it, and the
        totals must match exactly.
        """
        import torch

        def count(workers):
            ds = loader.MixDataset(mix, block_size=32, shuffle_shards=False)
            dl = torch.utils.data.DataLoader(ds, batch_size=1,
                                             num_workers=workers)
            return sum(1 for _ in dl)

        one, two = count(0), count(2)
        assert two == one, (
            f"1 worker yielded {one} sequences, 2 workers yielded {two}. "
            "Each worker must read only its own shards -- use "
            "self._my_paths(), not self.paths.")

    def test_one_worker_and_two_workers_see_the_same_data(self, loader, mix):
        """The split changes who reads what, not what gets read."""
        import torch
        def collect(workers):
            ds = loader.MixDataset(mix, block_size=32, shuffle_shards=False)
            dl = torch.utils.data.DataLoader(ds, batch_size=1,
                                             num_workers=workers)
            return {tuple(x[0].tolist()) for x, _ in dl}
        one, two = collect(0), collect(2)
        assert one == two, (
            f"1 worker saw {len(one)} sequences, 2 workers saw {len(two)}. "
            "Sharding must partition the data, not change it.")
