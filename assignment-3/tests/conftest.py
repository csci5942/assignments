"""Shared fixtures for the stage and loader tests.

One SparkSession for the whole session -- starting a JVM costs seconds,
and doing it per test would make these unbearable to run in a loop.
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "pipeline"), os.path.join(REPO, "data")):
    sys.path.insert(0, p)


@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession

    s = (SparkSession.builder.appName("a3_tests").master("local[2]")
         .config("spark.ui.enabled", "false")
         .config("spark.ui.showConsoleProgress", "false")
         .config("spark.sql.shuffle.partitions", "4")
         .getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


@pytest.fixture(scope="session")
def stages():
    """The Part 1 module under test.

    Set PIPELINE_MODULE=build_corpus_solution to run this suite against
    the instructor solution instead of the handout.
    """
    import importlib
    return importlib.import_module(
        os.environ.get("PIPELINE_MODULE", "build_corpus"))


@pytest.fixture(scope="session")
def loader():
    """The Part 2 module under test.

    Set LOADER_MODULE=loader_solution for the instructor version.
    """
    import importlib
    return importlib.import_module(
        os.environ.get("LOADER_MODULE", "loader"))


def make_wet(path, records):
    """Write a minimal but real WET file.

    `records` is a list of (url, body). The format is what Common Crawl
    actually emits: records separated by a `WARC/1.0` line, a header
    block, a blank line, then the page text.
    """
    import gzip

    parts = ["WARC/1.0\r\nWARC-Type: warcinfo\r\n"
             "WARC-Target-URI: \r\n\r\nfile metadata, no page body\r\n"]
    for url, body in records:
        parts.append(
            "WARC/1.0\r\n"
            "WARC-Type: conversion\r\n"
            f"WARC-Target-URI: {url}\r\n"
            "Content-Type: text/plain\r\n"
            f"Content-Length: {len(body)}\r\n"
            "\r\n"
            f"{body}\r\n")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("".join(parts))
    return str(path)


@pytest.fixture
def wet(tmp_path):
    """Build a WET file from (url, body) pairs."""
    def _make(records, name="test.warc.wet.gz"):
        return make_wet(tmp_path / name, records)
    return _make


# A page long enough to clear MIN_RECORD_CHARS, in plain English, with
# several lines. Used wherever a test needs a page that survives stage 1.
def english_page(extra=""):
    line = ("The quick brown fox jumps over the lazy dog and the dog "
            "sleeps in the warm sun of a long afternoon. ")
    return ("Welcome to the site\n"
            + (line + "\n") * 6
            + "Home About Contact\n"
            + extra)
