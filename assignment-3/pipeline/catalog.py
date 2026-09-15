#!/usr/bin/env python3
"""Catalog the corpus and compose training mixes. GIVEN -- read, do not write.

    python pipeline/catalog.py --corpus out/raw/corpus --out out/raw

Part 2 is about *designing* mixes, not about the plumbing that builds
them, so the plumbing is here. Read it: you need to know what columns a
mix can filter on and what a mix actually is on disk.

A catalog is one row per line of the corpus, carrying everything a query
can select on -- the quality features from Part 1 stage 4, the language
signal from stage 1, the source host, and which shard the line lives in.

A **mix** is a query plus a token budget. Not a folder you assemble by
hand: a predicate you can write down, re-run, and put in your report.
`write_mix` materialises one as its own Parquet shards plus a manifest.

Edit MIXES below. Everything else you can leave alone.
"""

import argparse
import json
import os
import sys
import time

from pyspark.sql import DataFrame, SparkSession, functions as F

# Assignment 1's 65-character vocabulary. At character level one token is
# one character, so n_tokens == n_chars; the column exists because Part 3
# may change tokenizer and then it will not.
A1_VOCAB = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# ---------------------------------------------------------------------
# YOUR MIXES.
#
# A mix is a SQL predicate over catalog columns:
#
#   doc_stopwords   how many of the six stopwords the source PAGE had
#                   (stage 1's crude language signal)
#   doc_lines       lines in the source page
#   doc_mean_line   mean characters per line in the source page
#   n_words         words in this line
#   n_chars         characters in this line
#   alpha_ratio     fraction of characters that are A-Za-z
#   upper_ratio     fraction of characters that are A-Z
#   stop_hits       how many stopwords this LINE contains, 0..6
#   ends_sentence   does the line end in . ! or ?
#   word_len_mean   mean word length in this line
#   rel_pos         where the line sits in its page, 0..1
#   host            the source domain
#
# Part 2 needs exactly one entry: `baseline`, the whole corpus with no
# filtering at all, cut to the budget. That is the mark Part 3 has to
# beat. In Part 3 you add mixes of your own.
# ---------------------------------------------------------------------
BUDGET = 50_000_000

MIXES = {
    # No filtering at all: your whole corpus cut to the budget.
    "baseline": "true",
    # The reference mix, and the mark Part 3 has to beat.
    "english_pages": "doc_stopwords >= 5",
    # Part 3: add your own here.
}

def build_catalog(corpus: DataFrame) -> DataFrame:
    """One row per line, plus where it lives and what it costs.

    Row count must equal the corpus row count -- this only adds columns.
    """
    return (corpus
            .withColumn("shard_id", F.regexp_extract(
                F.input_file_name(), r"([^/]+)\.parquet$", 1))
            .withColumn("n_tokens", F.col("n_chars").cast("long")))


def select_mix(catalog: DataFrame, where: str, target_tokens: int) -> DataFrame:
    """Filter to `where`, then cut down to `target_tokens`.

    The cut is a deterministic hash sample of whole lines, so it is
    reproducible and independent of partitioning. Sampling *lines* to hit
    a *token* target is not exact, though: the lines a hash bucket
    selects are not guaranteed to be of average length, and different
    filters leave pools with very different length distributions. A
    single pass lands anywhere from -4% to +2% of the budget, which is
    too loose for the equal-budget comparison Part 3 depends on -- so
    rescale the threshold by target/realized and try again. Two or three
    passes gets inside half a percent.
    """
    pool = catalog.filter(where)
    available = pool.agg(F.sum("n_tokens")).first()[0] or 0
    if available <= target_tokens:
        return pool

    bucket = F.abs(F.xxhash64(F.col("doc_id"))) % 1_000_000
    frac, best = target_tokens / available, None
    for _ in range(4):
        sel = pool.filter(bucket < int(frac * 1_000_000))
        got = sel.agg(F.sum("n_tokens")).first()[0] or 0
        if not got:
            break
        if best is None or abs(got - target_tokens) < abs(best[1] - target_tokens):
            best = (sel, got)
        if abs(got - target_tokens) / target_tokens <= 0.005:
            break
        frac *= target_tokens / got
    return best[0] if best else pool


def mix_fingerprint(mix: DataFrame) -> str:
    """A content hash of exactly which lines are in a mix.

    Order-independent: each doc_id is hashed to 64 bits and the values
    are XOR-folded, so repartitioning cannot change the answer while
    adding or dropping one line will. Two people reporting the same
    fingerprint trained on the same tokens.
    """
    v = mix.select(F.bit_xor(F.xxhash64(F.col("doc_id"))).alias("h")).first()["h"]
    return f"{(v or 0) & 0xFFFFFFFFFFFFFFFF:016x}"


def write_mix(mix: DataFrame, out_dir: str, name: str, shards: int = 16) -> dict:
    """Materialise a mix as its own Parquet shards, plus a manifest.

    The shards carry `text`, not token ids, so Part 3 can change
    tokenizer without re-running any Spark.
    """
    path = os.path.join(out_dir, f"mix_{name}")
    (mix.select("doc_id", "host", "text", "n_tokens")
        .repartition(shards)
        .write.mode("overwrite").parquet(path))
    man = {"name": name, "shards": shards, "path": path,
           "rows": mix.count(),
           "tokens": int(mix.agg(F.sum("n_tokens")).first()[0] or 0),
           "fingerprint": mix_fingerprint(mix)}
    with open(os.path.join(out_dir, f"mix_{name}.json"), "w") as f:
        json.dump(man, f, indent=2)
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True, help="Parquet from Part 1 stage 6")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--cores", default="*")
    ap.add_argument("--memory", default="8g")
    args = ap.parse_args()

    if not MIXES:
        print("MIXES is empty -- edit pipeline/catalog.py before running this.",
              file=sys.stderr)
        return 1
    if set(MIXES) == {"baseline"}:
        print("building the Part 2 baseline only; Part 3 adds mixes of your own")

    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (SparkSession.builder.appName("catalog")
             .master(f"local[{args.cores}]")
             .config("spark.driver.memory", args.memory)
             .config("spark.sql.shuffle.partitions", "64")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    corpus = spark.read.parquet(args.corpus)
    n_corpus = corpus.count()
    cat = build_catalog(corpus).cache()
    n_cat = cat.count()
    print(f"corpus {n_corpus:,} lines -> catalog {n_cat:,} rows")
    if n_cat != n_corpus:
        print(f"  WARNING: {n_cat - n_corpus:+,} rows. build_catalog only adds "
              "columns; it should not change the row count.")
    cat.write.mode("overwrite").parquet(os.path.join(args.out, "catalog"))

    corpus_tokens = cat.agg(F.sum("n_tokens")).first()[0] or 0
    report = {"catalog_rows": n_cat, "corpus_tokens": int(corpus_tokens),
              "budget": args.budget, "mixes": {}}

    for name, where in MIXES.items():
        t = time.time()
        pool = cat.filter(where)
        pool_tokens = pool.agg(F.sum("n_tokens")).first()[0] or 0
        mix = select_mix(cat, where, args.budget).cache()
        man = write_mix(mix, args.out, name)
        if man["tokens"] < args.budget * 0.95:
            print(f"  !! {name}: only {man['tokens']/1e6:.1f}M tokens, below the "
                  f"{args.budget/1e6:.0f}M budget -- this mix is too narrow to "
                  "compare fairly")
        report["mixes"][name] = {
            "where": where, "pool_tokens": int(pool_tokens),
            "pool_pct_of_corpus": round(100.0 * pool_tokens / corpus_tokens, 1)
            if corpus_tokens else 0,
            **man, "seconds": round(time.time() - t, 1)}
        print(f"  {name:14s} pool {pool_tokens/1e6:>7.1f}M tokens "
              f"({100.0*pool_tokens/max(1,corpus_tokens):>5.1f}% of corpus) "
              f"-> mix {man['tokens']/1e6:.1f}M, fp {man['fingerprint']}")
        mix.unpersist()

    with open(os.path.join(args.out, "catalog_stats.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}/catalog_stats.json")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
