# Assignment 3: Data Pipelines

CSCI 5942: AI Engineering, Fall 2026.
Due date and submission: see the course page (Gradescope).

In this assignment, you will use Spark to turn raw Common Crawl into a tagged corpus, train
assignment 1's `base` on that corpus **unfiltered** to get a baseline,
and then spend the rest of the assignment trying to beat your own
baseline by engineering the data — at an identical token budget.

## Part 0: Setup environment and corpus (nothing to submit)

**Java.** PySpark runs on a JVM. You need a Java 17+ runtime on your
PATH before anything else will work.

| where | how |
| :--- | :--- |
| Ubuntu / Debian | `sudo apt install openjdk-17-jre-headless` |
| macOS | `brew install openjdk@17`, then follow brew's symlink caveat |
| Windows | use WSL2 and follow the Ubuntu row, or install Temurin 17 from adoptium.net |

Confirm with `java -version`; it must print 17 or higher.

**Python 3.10+.** Check with `python3 --version`. The system Python on
Ubuntu 22.04 (3.10) is fine, so on most machines there is nothing to do
here.

Then, from the `assignment-3/` repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python check_env.py
```

`check_env.py` prints a pass/fail line per requirement and, for
anything that fails, what to do about it. 

Nothing else is needed. This repository is self-contained: assignment
1's model and vocabulary are vendored at `model.py` and
`data/vocab.json`, and assignment 2's evaluation harness and its four
corpora are under `eval/`. The corpora are byte-identical to A2's, so
your numbers here are directly comparable to the ones you reported
there.

Once the environment check passes:

```bash
python -m pytest tests/ -q        
```

`test_env` should pass immediately. Every other failure should be a
`NotImplementedError` from code you have not written. 

### Getting the corpus

We use **Common Crawl**, the open web crawl, in WET format: plain text
already extracted from HTML, one record per page, with the source URL.

```bash
python data/fetch.py --files 1  --out data/dev/    # ~80MB gz, for development
python data/fetch.py --files 5  --out data/raw/    # ~380MB gz, the graded set
```

**Develop against `data/dev/`.** 
`build_corpus.py` takes `--limit N` to use only N records and `--stages N` to stop after
stage N, so while working on the chunker run
`--limit 200 --stages 2 --peek 5` to iterate quickly:

```bash
python pipeline/build_corpus.py --input data/dev --out out/dev \
       --limit 200 --stages 2 --peek 5
```

Once every stage is right, build the graded corpus. This is the command
whose output Parts 2 and 3 consume.

```bash
python pipeline/build_corpus.py --input data/raw --out out/raw
```

That writes the Parquet shards to `out/raw/corpus` and the per-stage
row, byte and timing table to `out/raw/funnel.json` -- both are
deliverables.


## Part 1: Build the corpus with Spark

`pipeline/build_corpus.py` has the stage skeletons. Implement, in order:

1. **Ingest.** Reading the WET records and dropping the eval domain are
   both **given**: `read_wet` hands you one row per page with its
   source host. You must tag each document. Label the doc with
   (`doc_stopwords`) and the page's shape (`doc_lines`,
   `doc_mean_line`). This stage removes no rows.
2. **Chunk.** Split each page into lines. One row per line.
3. **Normalize.** Collapse whitespace, restrict to assignment 1's
   65-character vocabulary.
4. **Quality features.** Per line: character count, word count,
   alphabetic ratio, uppercase ratio, stopword hits.
5. **Exact dedup.** Hash the text, keep one per hash.
6. **Write.** Parquet shards.

**Part 1 tags. It does not filter.** The goal of Part 1
is to label your corpus with features that *may* be used
as quality signals when you build your training mix.
You will decide what counts as good in Part 2,
and measure it in part 3.

## Part 2: Feed the GPU, and get your baseline

`pipeline/catalog.py` is **given**. It turns
your corpus into a catalog (one row per line, carrying everything a
query can filter on) and materialises a *mix* as its own Parquet shards
plus a manifest. A mix is a **query plus a token budget**:

```python
MIXES = {
    "baseline": "true",              # your whole corpus, no filtering
    "english_pages": "doc_stopwords >= 5",   # the reference mix, given
}
```

Two mixes come with the file. `baseline` is the floor: no filtering at
all. `english_pages` keeps only lines from pages carrying at least five
of the six stopwords, a rough proxy of english pages. 

### The loader

`data/loader.py` is given except for one function: **`MixDataset.__iter__`**.
Read the rest of the file first, and then implement `__iter__`,
which composes your mix into a stream of `(input, target)` pairs.

```bash
python -m pytest tests/test_loader.py -q
```

### Your baseline, and the mark to beat

`catalog.py` builds both mixes in one pass. Train and score each:

```bash
python pipeline/catalog.py --corpus out/raw/corpus --out out/raw
python train.py --mix out/raw/mix_baseline --name baseline
python eval/eval.py --run out/baseline --name baseline --no-opik
python train.py --mix out/raw/mix_english_pages --name english_pages
python eval/eval.py --run out/english_pages --name english_pages --no-opik
```

Training runs 3000 iterations by default: about 6 minutes on an L4,
roughly 25 on a T4. Scoring adds a couple of minutes.

Record your evaluation results. For reference, here is one
example training run on the provided mixes.

| mix | tokens | wikipedia | tinystories | shakespeare | python-code |
|---|---:|---:|---:|---:|---:|
| `baseline` | 49.9M | 1.86 | 1.94 | 2.81 | 3.05 |
| `english_pages` | 50.1M | **1.83** | 1.90 | 2.82 | 3.25 |

While you will not obtain the exact same results,
if you are very far off, sanity-check your work before moving on.

## Part 3: Beat it

 **Engineer the data so the model gets better.**

In part 1, you've already calculated various features over the data.
Use them to design filters over the catalog.
Build a mix, train, score, and repeat.
Your goal is to beat your Part 2 reference mixes on **wikipedia** at the *same
token budget*.

Once you've beaten the reference mixes in Part 2 (which should be relatively
straightforward), continue engineering your data to achieve the best
performance on wikipedia.
For reference, one filter we tried reaches **1.50** on wikipedia. 

Concretely, **add at least one custom feature** to the `build_corpus.py` pipeline
by adding an additional stage.
This feature does not have to be used in the best-performing mix,
but try to design something sensible and evaluate how well it performs.


*(Optional)*: Notice that Stage 3 deleted every character outside of 
A1's vocab, which is a significant amount of text. As an optional exercise,
remove this filter and augment your model and data loader to be able
to train on this higher-quality data!

## Deliverables

Submit your code plus one PDF.

**Code and artifacts**

- `pipeline/build_corpus.py`, `data/loader.py`
- every mix definition with its token count
- `funnel.json` and your Spark stage timings
- `eval/results/*.json`, one per run, **including the runs that failed**

**Report.**:

1. **Data Analysis.** Provide an analysis over your Spark pipeline.
   In one or two paragraphs, discuss your observations about the stages.
   Which stages filtered out the most data? What does that say about your dataset?
   Look at the statistical metrics of your feature columns (e.g., `df.describe()`,
   or `df.show()`), and make some observations about your dataset and features
   based on these values.

2. **The grid.** Show the evaluation grid (across the four corpora) over at least
   four mixes: the `baseline`, `english_pages`, your best performing mix,
   and a mix that you tried that didn't work.

3. **Analyze.** Describe your best mix on wikipedia. 
   What was it keeping or discarding that the baseline was not? Show a few example lines highlighting what it 
   either kept or dropped compared to the baseline, and analyze why
   that would perform better on wikipedia.
   Analyze the other evaluation corpora, and discuss how this mix performed
   on the other datasets.

4. **What did not work.** Discuss the mix that you tried that failed,
   and explain why you think it didn't work as well.

5. **Anything you changed in Part 1** — Discuss the new feature(s) you added, or a
   different vocabulary. Say why the existing signals in the pipeline
   were not enough, and your intuition on why this feature(s) may be helpful.

## If you have not used Spark before

See the following for further references:

- [`pyspark.sql.functions`](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html)
- [DataFrame API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html)
- [Spark SQL programming guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [Quickstart: DataFrame](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_df.html)

## Layout

```
requirements.txt          pinned dependencies
check_env.py              environment check with actionable failures
model.py                  assignment 1's transformer (vendored, given)
train.py                  assignment 1's training loop, loader-backed (given)
data/vocab.json           assignment 1's 65-character vocabulary
data/fetch.py             pull Common Crawl WET files (given)
data/loader.py            streaming IterableDataset -- YOURS (Part 2)
pipeline/build_corpus.py  the Spark pipeline -- YOURS (Part 1)
pipeline/catalog.py       catalog + mix composition (given; edit MIXES)
eval/eval.py              score a checkpoint on four corpora (vendored)
eval/protocol.py          the fixed loss protocol (vendored)
eval/corpora/             the four eval corpora (fixed, do not edit)
tests/test_env.py         environment checks (should pass immediately)
tests/test_stages.py      Part 1 stage tests
tests/test_loader.py      Part 2 loader tests
```
