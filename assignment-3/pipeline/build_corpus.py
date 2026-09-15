#!/usr/bin/env python3
"""Build a clean, deduplicated training corpus from Common Crawl.

Part 1 of Assignment 3. Six stages, each yours to write from its
docstring. Parsing the WET files is given -- `read_wet` -- so stage 1
is only the tagging. You do not need to change anything below the
`stages` section.

    python pipeline/build_corpus.py --input data/dev --out out/dev

Each stage's docstring gives the columns it receives and the columns it
must return. If you have not used Spark before, read the notes at the
top of the stages section first.

Work on `data/dev` (one WET file) until the funnel table looks right,
then run the graded five-file set. Sanity check the row count after
every stage. If a stage is wrong, it will not crash, but it will return
the wrong corpus.
"""

import argparse
import json
import os
import re
import sys
import time

from pyspark.sql import DataFrame, SparkSession, functions as F

STOPWORDS = ["the", "and", "of", "to", "in", "a"]

# --- ingest (stage 1) ------------------------------------------------
# WARC records are separated by this literal line, so it doubles as a
# record delimiter for spark.read.text(lineSep=...).
WARC_DELIM = "WARC/1.0"
MIN_RECORD_CHARS = 500

# Filter out the eval domain (wikipedia)
CONTAM = ("wikipedia.org", "wikimedia.org", "wikisource.org", "wiktionary.org")

# Assignment 1's 65-character vocabulary.
A1_VOCAB = "\n !$&\',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

LINE_SPLIT = r"\r?\n"


def _todo(what: str):
    """Placeholder for a line you need to write. Replace the whole call."""
    raise NotImplementedError(what)


# =====================================================================
# stages
# =====================================================================
# If you have not used Spark before, here is a quick introduction.
#
#   A DataFrame is a lazy table. Calling .filter(), .withColumn() or
#   .select() does not compute anything; it records what to do. Work
#   happens only on an "action" like .count() or .write. The harness at
#   the bottom calls .count() after each stage, which is why you see
#   per-stage timings at all.
#
#   You never write loops over rows. You build *column expressions* out
#   of `F` (imported above as `pyspark.sql.functions`) and hand them to
#   the DataFrame. `F.col("text")` refers to a column; `F.length(...)`,
#   `F.lower(...)`, `F.split(...)` transform one.
#
#   df.withColumn("new", expr)   add or replace a column
#   df.select("a", "b")          keep these columns
#   df.filter(condition)         keep rows where condition is true
#   df.groupBy("k").agg(...)     aggregate per key
#   df.join(other, "k", how)     join; `how` can be "inner",
#                                "left_semi" (keep left rows with a
#                                match), "left_anti" (keep left rows
#                                *without* one)
#
#   Combine conditions with & and |, and parenthesize every operand:
#   (F.col("a") > 1) & (F.col("b") < 2).
#
# Full function list: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html
#
# --- looking at your data --------------------------------------------
#
# `print(df)` prints only the schema, because nothing has been computed
# yet. To see rows you need an action:
#
#   df.show(5, truncate=80)      5 rows as a table
#   df.show(3, vertical=True)    one field per line; better for long text
#   df.printSchema()             column names and types
#   df.count()                   row count
#
# You can call these inside a stage while you work -- build the frame,
# `.show()` it, then return it. Each call runs a real Spark job, so
# `.cache()` the frame first if you are going to look at it repeatedly,
# and take them out when you are done.
#
# Without editing anything, this runs the pipeline on 5 pages and prints
# the schema and 5 rows after every stage:
#
#   python pipeline/build_corpus.py --input data/dev --out /tmp/x \
#          --limit 5 --peek 5
#
# To poke at stages interactively, import them in a REPL:
#
#   >>> import sys; sys.path.insert(0, "pipeline")
#   >>> from pyspark.sql import SparkSession, functions as F
#   >>> import build_corpus as bc
#   >>> spark = SparkSession.builder.master("local[4]").getOrCreate()
#   >>> pages = bc.stage_01_ingest(spark, "data/dev").limit(3).cache()
#   >>> lines = bc.stage_02_chunk(pages)
#   >>> lines.show(5, truncate=70)


def read_wet(spark: SparkSession, path: str) -> DataFrame:
    """Parse WET files into one row per web page. GIVEN -- you do not write this.

    In:  `path`, a glob of `.warc.wet.gz` files.
    Out: DataFrame[host: string, text: string], one row per page whose
         text is at least MIN_RECORD_CHARS long.
    """
    contam = "|".join(re.escape(h) for h in CONTAM)
    return (spark.read.text(path, lineSep=WARC_DELIM)
            .withColumn("uri", F.regexp_extract(
                "value", r"WARC-Target-URI:\s*(\S+)", 1))
            .withColumn("host", F.lower(F.regexp_extract(
                F.col("uri"), r"https?://([^/]+)", 1)))
            # (?s) so `.` spans newlines: the body is everything after
            # the blank line that ends the WARC header block
            .withColumn("text", F.regexp_extract(
                "value", r"(?s)\r?\n\r?\n(.*)", 1))
            .filter(F.length("host") > 0)
            .filter(F.length("text") >= MIN_RECORD_CHARS)
            .filter(~F.col("host").rlike(contam))     # decontamination
            .select("host", "text"))


def stage_01_ingest(spark: SparkSession, path: str) -> DataFrame:
    """Tag each page: a language signal and the page's shape.

    In:  `path`, a glob of `.warc.wet.gz` files. `read_wet` above does
         the parsing and the decontamination, and hands you
         DataFrame[host, text], one row per page -- that part is given.
    Out: DataFrame[host: string, text: string, doc_stopwords: int,
         doc_lines: int, doc_mean_line: double]

    Your tasks:
    1. Count how many of `STOPWORDS` appear in the page as `doc_stopwords`.
       A page with most of the six common English function words is probably
       English, and will be a feature used in Part 2.
    2. **Tag the page's shape.** `doc_lines` is how many lines the page has and
       `doc_mean_line` the mean characters per line. 

    For `doc_stopwords`, `F.instr(haystack, needle)` returns a 1-based
    position or 0 if absent, so `(F.instr(...) > 0).cast("int")` 
    may be helpful in counting how many of the six stopwords appear.
    However, be sure to pad the needle with spaces, 
    or else you may see extraneous matches (e.g., " a " instead of "a").
    """
    pages = read_wet(spark, path)
    # TODO: implement this stage.
    raise NotImplementedError("stage not implemented")


def stage_02_chunk(pages: DataFrame) -> DataFrame:
    r"""Split each page into lines.

    In:  the DataFrame from stage 1  -- one row per page
    Out: its page-level columns plus line_ix: int and text: string,
         one row per non-blank line.

    1. Split each page's text into lines, where `line_ix` contains
    the line's 0-based index in the page and `text` has the line's content.
    2. Filter out lines that are empty or whitespace-only.

    Hint: `LINE_SPLIT` may be helpful.
    """
    # TODO: implement this stage.
    raise NotImplementedError("stage not implemented")


def stage_03_normalize(lines: DataFrame) -> DataFrame:
    """Clean up each line and restrict it to assignment 1's vocabulary.

    In:  DataFrame[host, doc_stopwords, line_ix, text]
    Out: the same schema, fewer rows (lines that normalize to nothing).

    1. Collapse runs of whitespace to single spaces and strip the ends.
    2. Delete every character outside `A1_VOCAB`. 

    **Order matters** The vocabulary contains space and newline but 
    not tab or carriage return, so be sure to collapse whitespace *before* 
    restricting the vocabulary.

    Drop anything that ends up empty.
    """
    # TODO: implement this stage.
    raise NotImplementedError("stage not implemented")


def stage_04_features(paras: DataFrame) -> DataFrame:
    """Compute the per-line quality features.

    In:  the DataFrame from stage 3
    Out: the same rows plus the following 
         feature columns. Row count must not change --
         this stage only adds columns.

         n_chars       int     characters in the line
         n_words       int     whitespace-separated tokens
         alpha_ratio   double  fraction of characters that are A-Za-z
         upper_ratio   double  fraction of characters that are A-Z
         stop_hits     int     how many of STOPWORDS appear in the line
         ends_sentence boolean does it end in . ! or ?
            (hint: prose does, but navigation labels and buttons do not)
         word_len_mean double  mean word length
            (hint: URLs and identifiers are often longer than ordinary prose)
         rel_pos       double  where the line sits in its page, 0..1.
            (hint: Headers and footers cluster at the ends)
    """
    # TODO: implement this stage.
    raise NotImplementedError("stage not implemented")


def stage_05_exact_dedup(paras: DataFrame) -> DataFrame:
    """Deduplicate byte-identical lines.

    In:  the DataFrame from stage 4.
    Out: the same columns plus `doc_id: string`, which is a hash (e.g., SHA-256)
        of the line's text, with at most one row per distinct `text`
    """
    # TODO: implement this stage.
    raise NotImplementedError("stage not implemented")


def stage_06_write(paras: DataFrame, out_dir: str, shards: int) -> DataFrame:
    """Write the finished corpus to Parquet.

    In:  the DataFrame from stage 5; `out_dir`; `shards`, how many
         output files to write.
    Out: return the DataFrame unchanged so the harness can count it.
         Side effect: Parquet under `<out_dir>/corpus`.
    """
    # TODO: implement this stage.
    raise NotImplementedError("stage not implemented")


# =====================================================================
# harness (given -- you do not need to change anything below)
# =====================================================================
def measure(df: DataFrame, name: str, funnel: list, t0: float,
            peek: int = 0) -> DataFrame:
    """Materialize `df`, record rows/bytes/elapsed, return it cached.

    Counting at every stage costs time. It is also the only way to get
    the funnel table Part 1 asks for, and the only way to notice a stage
    that silently dropped everything.

    With --peek N, prints N rows and the schema after each stage. That
    is usually the fastest way to see what your transformation actually
    did, and it is why the frame is cached first: showing rows would
    otherwise recompute the whole stage.
    """
    df = df.cache()
    rows = df.count()
    col = "text" if "text" in df.columns else None
    nbytes = (df.agg(F.sum(F.octet_length(F.col(col)))).first()[0] or 0) if col else 0
    dt = time.time() - t0
    funnel.append({"stage": name, "rows": rows, "bytes": int(nbytes),
                   "seconds": round(dt, 1)})
    print(f"  {name:24s} {dt:7.1f}s  {rows:>12,} rows  {nbytes/1e6:>9.1f} MB",
          flush=True)
    if peek:
        print(f"\n--- {name}: schema ---")
        df.printSchema()
        print(f"--- {name}: first {peek} rows ---")
        df.show(peek, truncate=90, vertical=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True,
                    help="glob of WET files, e.g. 'data/dev/*.warc.wet.gz'")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--cores", default="*", help="local[N] parallelism")
    ap.add_argument("--memory", default="8g")
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--partitions", type=int, default=64)
    ap.add_argument("--peek", type=int, default=0, metavar="N",
                    help="print the schema and N rows after every stage. "
                         "The fastest way to see what a stage actually did")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="use only the first N pages, for quick iteration")
    ap.add_argument("--stages", type=int, default=6, metavar="N",
                    choices=range(1, 7),
                    help="run only stages 1..N and stop. Use this while "
                         "working on one stage")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    spark = (SparkSession.builder.appName("build_corpus")
             .master(f"local[{args.cores}]")
             .config("spark.driver.memory", args.memory)
             .config("spark.sql.shuffle.partitions", args.partitions)
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    funnel: list[dict] = []
    wall = time.time()
    print(f"building corpus from {args.input}")

    # Stage 1 is separate because it takes the path rather than a frame,
    # and --limit applies to pages, before anything is chunked.
    t = time.time()
    pages = stage_01_ingest(spark, args.input)
    if args.limit:
        pages = pages.limit(args.limit)
    df = measure(pages.repartition(args.partitions), "1 ingest", funnel, t,
                 args.peek)

    rest = [
        ("2 chunk", stage_02_chunk),
        ("3 normalize", stage_03_normalize),
        ("4 features", stage_04_features),
        ("5 exact dedup", stage_05_exact_dedup),
        ("6 write", lambda d: stage_06_write(d, args.out, args.shards)),
    ]
    for name, fn in rest:
        if int(name.split()[0]) > args.stages:
            break
        t = time.time()
        df = measure(fn(df), name, funnel, t, args.peek)

    if args.stages < 6:
        print(f"\n  stopped after stage {args.stages} (--stages); "
              "no Parquet written")

    total = time.time() - wall
    base = (funnel[1]["rows"] if len(funnel) > 1 else funnel[0]["rows"]) or 1
    for row in funnel:
        row["pct_of_lines"] = round(100.0 * row["rows"] / base, 2)
    report = {"input": args.input, "total_seconds": round(total, 1),
              "books": funnel[0]["rows"], "funnel": funnel}
    with open(os.path.join(args.out, "funnel.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  {'stage':24s} {'rows':>14s} {'% of lines':>11s} {'sec':>8s}")
    for row in funnel:
        print(f"  {row['stage']:24s} {row['rows']:>14,} "
              f"{row['pct_of_lines']:>10.2f}% {row['seconds']:>8.1f}")
    print(f"\n  total {total:.1f}s -> {args.out}/funnel.json")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
