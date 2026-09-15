"""Stage tests for Part 1.

    python -m pytest tests/test_stages.py -q

Every fixture is small enough to work out by hand, which is the point:
you should be able to say what the right answer is before running the
test. Each stage's tests encode its docstring's contract, plus the
specific ways that stage fails *silently* on web crawl.

Until you implement a stage its tests fail with NotImplementedError.
Work down the file in order.
"""

import os

import pytest
from pyspark.sql import functions as F

from conftest import english_page

GOOD = ("It is a truth universally acknowledged that a single man in "
        "possession of a good fortune must be in want of a wife however "
        "little known the feelings or views of such a man may be on his "
        "first entering a neighbourhood this truth is so well fixed in "
        "the minds of the surrounding families that he is considered as "
        "the rightful property of some one or other of their daughters")

LINE_COLS = ["host", "doc_stopwords", "doc_lines", "doc_mean_line",
             "line_ix", "text"]


def lines_df(spark, texts):
    """A frame shaped like stage 2's output."""
    n = max(1, len(texts))
    return spark.createDataFrame(
        [("h.com", 5, n, 80.0, i, t) for i, t in enumerate(texts)], LINE_COLS)


# =====================================================================
# stage 1, given half: read_wet parses and decontaminates
# =====================================================================
# These cover code students are handed rather than write. They should
# pass on a fresh checkout: if one fails, the handout itself is broken.
class TestReadWet:
    def test_skips_the_file_metadata_record(self, spark, stages, wet):
        """The first record of every WET file is the file's own warcinfo.

        It has no Target-URI and no page body. Counting it gives one
        phantom 'page' whose text is a WARC header block.
        """
        path = wet([("http://a.com/1", english_page())])
        assert stages.read_wet(spark, path).count() == 1

    def test_host_extracted_and_lowercased(self, spark, stages, wet):
        path = wet([("https://Example.COM/some/page?q=1", english_page())])
        assert stages.read_wet(spark, path).first()["host"] == "example.com"

    def test_body_is_page_text_not_headers(self, spark, stages, wet):
        path = wet([("http://a.com/1", english_page())])
        text = stages.read_wet(spark, path).first()["text"]
        assert "WARC-Target-URI" not in text, \
            "the body starts after the blank line that ends the header block"
        assert "quick brown fox" in text

    def test_drops_tiny_pages(self, spark, stages, wet):
        path = wet([("http://a.com/1", "too short"),
                    ("http://b.com/2", english_page())])
        assert stages.read_wet(spark, path).count() == 1

    def test_decontamination_drops_wikipedia(self, spark, stages, wet):
        """Common Crawl crawls Wikipedia, and Wikipedia is the eval set.

        Training on it scores the model against its own training data.
        This drop is given, and no mix may opt those hosts back in.
        """
        path = wet([("https://en.wikipedia.org/wiki/Cat", english_page()),
                    ("https://fr.wikipedia.org/wiki/Chat", english_page()),
                    ("http://keepme.com/x", english_page())])
        rows = stages.read_wet(spark, path).collect()
        assert len(rows) == 1, "both wikipedia hosts must be dropped"
        assert rows[0]["host"] == "keepme.com"

    def test_decontamination_does_not_overmatch(self, spark, stages, wet):
        """Dots in a hostname are regex wildcards unless escaped.

        This host is legitimate and must survive.
        """
        path = wet([("http://notwikipediaXorg.com/x", english_page())])
        assert stages.read_wet(spark, path).count() == 1


# =====================================================================
# stage 1: tag each page
# =====================================================================
class TestIngest:
    def test_one_row_per_page(self, spark, stages, wet):
        path = wet([("http://a.com/1", english_page()),
                    ("http://b.com/2", english_page())])
        df = stages.stage_01_ingest(spark, path)
        assert df.count() == 2
        assert {"host", "text", "doc_stopwords", "doc_lines",
                "doc_mean_line"} <= set(df.columns)

    def test_tagging_removes_no_rows(self, spark, stages, wet):
        """Stage 1 only adds columns. read_wet already did the dropping."""
        path = wet([("http://a.com/1", english_page()),
                    ("http://b.com/2", english_page()),
                    ("http://c.com/3", english_page())])
        assert (stages.stage_01_ingest(spark, path).count()
                == stages.read_wet(spark, path).count())

    def test_page_shape_is_tagged(self, spark, stages, wet):
        """doc_lines counts lines; doc_mean_line is chars per line.

        A navigation-heavy page has many short lines, an article fewer
        long ones, and a Part 2 mix can select on the difference.
        """
        nav = "\n".join(["Home", "About", "Contact", "Login"] * 40)
        path = wet([("http://nav.com/1", nav),
                    ("http://art.com/2", english_page())])
        rows = {r["host"]: r for r in
                stages.stage_01_ingest(spark, path).collect()}
        nav_row = rows["nav.com"]
        assert nav_row["doc_lines"] == len(nav_row["text"].split("\n"))
        assert nav_row["doc_lines"] >= 160
        assert rows["nav.com"]["doc_mean_line"] < \
            rows["art.com"]["doc_mean_line"], \
            "navigation lines are shorter than prose lines"

    def test_language_is_tagged_not_filtered(self, spark, stages, wet):
        """Stage 1 tags the language signal; it must not act on it.

        Part 2 decides whether to filter and Part 3 measures whether that
        was right. Dropping non-English here makes the question
        unanswerable.
        """
        french = ("Bonjour le monde ceci est une page en francais qui "
                  "parle de choses interessantes pour les lecteurs. ") * 6
        path = wet([("http://fr.com/1", french),
                    ("http://en.com/2", english_page())])
        rows = {r["host"]: r["doc_stopwords"]
                for r in stages.stage_01_ingest(spark, path).collect()}
        assert len(rows) == 2, "the French page must survive stage 1"
        assert rows["en.com"] > rows["fr.com"], \
            "the English page should score higher on the stopword signal"


# =====================================================================
# stage 2: chunk pages into lines
# =====================================================================
class TestChunk:
    def test_splits_on_newlines(self, spark, stages, wet):
        path = wet([("http://a.com/1", english_page())])
        out = stages.stage_02_chunk(stages.stage_01_ingest(spark, path))
        assert out.count() == 8, (
            "the fixture page has 8 non-blank lines. Web text is one "
            "visible line per line -- do not split on blank lines.")
        assert {"line_ix", "text"} <= set(out.columns)

    def test_drops_blank_and_whitespace_lines(self, spark, stages, wet):
        body = ("First line here with plenty of words.\n\n   \n\t\n"
                "Second line here with plenty of words.\n") + english_page()
        path = wet([("http://a.com/1", body)])
        texts = [r["text"] for r in
                 stages.stage_02_chunk(stages.stage_01_ingest(spark, path)).collect()]
        assert all(t.strip() for t in texts)
        assert "\r" not in texts, (
            "a line holding only a carriage return survives F.trim, which "
            "strips ASCII space only")

    def test_line_ix_is_zero_based(self, spark, stages, wet):
        path = wet([("http://a.com/1", english_page())])
        rows = sorted(
            stages.stage_02_chunk(stages.stage_01_ingest(spark, path)).collect(),
            key=lambda r: r["line_ix"])
        assert rows[0]["line_ix"] == 0
        assert "Welcome to the site" in rows[0]["text"]

    def test_carries_page_tags_down(self, spark, stages, wet):
        """Every line needs its page's language signal, or Part 2 cannot
        filter on it."""
        path = wet([("http://a.com/1", english_page())])
        out = stages.stage_02_chunk(stages.stage_01_ingest(spark, path))
        assert {"host", "doc_stopwords", "doc_lines",
                "doc_mean_line"} <= set(out.columns)
        assert out.filter(F.col("doc_stopwords").isNull()).count() == 0


# =====================================================================
# stage 3: normalize and restrict to the vocabulary
# =====================================================================
class TestNormalize:
    def _one(self, spark, stages, text):
        rows = stages.stage_03_normalize(lines_df(spark, [text])).collect()
        return rows[0]["text"] if rows else None

    def test_collapses_whitespace(self, spark, stages):
        assert self._one(spark, stages, "one   two\tthree") == "one two three"

    def test_trims_ends(self, spark, stages):
        assert self._one(spark, stages, "   padded   ") == "padded"

    def test_deletes_out_of_vocabulary_characters(self, spark, stages):
        """The model has 65 tokens; anything else cannot be represented.

        Most of the web is not English, so this deletes a lot -- roughly
        a quarter of all characters on raw crawl. Report the number.
        """
        assert self._one(spark, stages, "café naïve") == "cafe naive"
        assert self._one(spark, stages, "café naïve") == "caf nave"

    def test_deletes_non_latin_entirely(self, spark, stages):
        assert self._one(spark, stages, "你好世界") is None, \
            "a line with no representable characters must be dropped"

    def test_keeps_vocabulary_punctuation(self, spark, stages):
        assert self._one(spark, stages, "Don't stop! $3 & more?") == \
            "Don't stop! $3 & more?"

    def test_drops_digits_other_than_three(self, spark, stages):
        """A quirk of inheriting assignment 1's vocabulary: tiny-shakespeare
        contained '3' and no other digit, so every other numeral is gone.
        On web text -- dates, prices, statistics -- that is substantial."""
        assert self._one(spark, stages, "in 2024 the price was 1250") == \
            "in the price was"

    def test_row_count_only_falls(self, spark, stages):
        df = lines_df(spark, ["alpha", "beta  gamma", "delta"])
        assert stages.stage_03_normalize(df).count() == 3


# =====================================================================
# stage 4: quality features
# =====================================================================
class TestFeatures:
    def test_adds_columns_without_dropping_rows(self, spark, stages):
        out = stages.stage_04_features(
            lines_df(spark, [GOOD, "short", "another one here"]))
        assert out.count() == 3, "stage 4 adds columns; it must not filter"
        assert {"n_chars", "n_words", "alpha_ratio", "upper_ratio",
                "stop_hits", "ends_sentence", "word_len_mean",
                "rel_pos"} <= set(out.columns)

    def test_counts(self, spark, stages):
        row = stages.stage_04_features(lines_df(spark, ["the cat sat"])).first()
        assert row["n_chars"] == 11
        assert row["n_words"] == 3

    def test_ratios(self, spark, stages):
        row = stages.stage_04_features(lines_df(spark, ["AB12"])).first()
        assert row["alpha_ratio"] == pytest.approx(0.5)
        assert row["upper_ratio"] == pytest.approx(0.5)

    def test_stopwords_not_matched_as_substrings(self, spark, stages):
        """'there' contains 'the', 'brander' contains 'and'. Neither counts.

        Match whole words. Searching for the bare substring scores every
        line at maximum and your Part 2 thresholds stop discriminating.
        """
        row = stages.stage_04_features(
            lines_df(spark, ["there brander offer intone about"])).first()
        assert row["stop_hits"] == 0

    def test_stopwords_counted_when_present(self, spark, stages):
        row = stages.stage_04_features(lines_df(spark, [" the and of x "])).first()
        assert row["stop_hits"] == 3

    def test_ends_sentence(self, spark, stages):
        """Prose ends in terminal punctuation. Navigation labels do not."""
        rows = {r["text"]: r["ends_sentence"] for r in stages.stage_04_features(
            lines_df(spark, ["This is a sentence.", "Home About Contact",
                             "Really?", "Wow!"])).collect()}
        assert rows["This is a sentence."] and rows["Really?"] and rows["Wow!"]
        assert not rows["Home About Contact"]

    def test_word_len_mean(self, spark, stages):
        # "ab cd ef" -> 6 non-space chars over 3 words
        row = stages.stage_04_features(lines_df(spark, ["ab cd ef"])).first()
        assert row["word_len_mean"] == pytest.approx(2.0)

    def test_rel_pos_spans_zero_to_one(self, spark, stages):
        rows = sorted(stages.stage_04_features(
            lines_df(spark, ["a", "b", "c", "d", "e"])).collect(),
            key=lambda r: r["line_ix"])
        assert rows[0]["rel_pos"] == pytest.approx(0.0)
        assert rows[-1]["rel_pos"] == pytest.approx(1.0)


# =====================================================================
# stage 5: exact dedup
# =====================================================================
class TestExactDedup:
    def _rows(self, spark, stages, texts):
        return stages.stage_05_exact_dedup(
            stages.stage_04_features(lines_df(spark, texts)))

    def test_collapses_identical_lines(self, spark, stages):
        """Web boilerplate repeats verbatim across pages constantly --
        on a raw crawl this stage removes about half of it."""
        assert self._rows(spark, stages,
                          [GOOD, GOOD, GOOD + " Indeed."]).count() == 2

    def test_adds_doc_id(self, spark, stages):
        out = self._rows(spark, stages, [GOOD])
        assert "doc_id" in out.columns and out.first()["doc_id"]

    def test_doc_id_is_a_content_hash(self, spark, stages):
        """Same text, same id, on any run and any partitioning.

        A row number from monotonically_increasing_id() depends on
        partitioning, so it changes between runs and the dedup itself
        stops being reproducible.
        """
        a = self._rows(spark, stages, [GOOD]).first()["doc_id"]
        b = (self._rows(spark, stages, ["x y z", GOOD])
             .filter(F.col("text") == GOOD).first()["doc_id"])
        assert a == b


# =====================================================================
# stage 6: write
# =====================================================================
# a second line, unrelated to GOOD, so the write tests have two rows
FAR = ("The quick brown fox jumps over the lazy dog while a diligent bee "
       "gathers nectar from the blossoms in the meadow beyond the old wall")


class TestWrite:
    def _frame(self, spark, stages, texts):
        return stages.stage_05_exact_dedup(
            stages.stage_04_features(lines_df(spark, texts)))

    def test_round_trips_through_parquet(self, spark, stages, tmp_path):
        df = self._frame(spark, stages, [GOOD, FAR])
        stages.stage_06_write(df, str(tmp_path), shards=2)
        back = spark.read.parquet(os.path.join(str(tmp_path), "corpus"))
        assert back.count() == df.count()
        assert {"host", "text", "n_words", "doc_id",
                "doc_stopwords"} <= set(back.columns)

    def test_returns_the_frame(self, spark, stages, tmp_path):
        df = self._frame(spark, stages, [GOOD])
        assert stages.stage_06_write(df, str(tmp_path), shards=1) is not None

    def test_overwrites_on_rerun(self, spark, stages, tmp_path):
        df = self._frame(spark, stages, [GOOD, FAR])
        stages.stage_06_write(df, str(tmp_path), shards=1)
        stages.stage_06_write(df, str(tmp_path), shards=1)
        back = spark.read.parquet(os.path.join(str(tmp_path), "corpus"))
        assert back.count() == 2
