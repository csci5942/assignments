"""Environment tests for Assignment 3.

    python -m pytest tests/test_env.py -q

These must all pass before you start Part 1. `python check_env.py`
covers the same ground with friendlier output and some advisory
checks; this file is the version that gates the graded test run.
"""

import os
import re
import shutil
import subprocess
import sys

import pytest


def test_python_version():
    assert sys.version_info >= (3, 10), (
        f"Python 3.10+ required, found {sys.version.split()[0]}. "
        "Rebuild your venv with a newer interpreter."
    )


def test_java_17_available():
    """PySpark needs a JVM. pip cannot install one."""
    assert shutil.which("java") is not None, (
        "No java on PATH. PySpark needs a Java 17+ runtime:\n"
        "  Linux:  sudo apt install openjdk-17-jre-headless\n"
        "  macOS:  brew install openjdk@17\n"
        "  Windows: install Temurin 17 from adoptium.net, or use WSL2"
    )
    out = subprocess.run(["java", "-version"], capture_output=True,
                         text=True, timeout=30)
    text = (out.stderr or "") + (out.stdout or "")
    m = re.search(r'version "?(\d+)', text)
    assert m, f"could not parse java version from: {text!r}"
    assert int(m.group(1)) >= 17, f"Java 17+ required, found {m.group(1)}"


@pytest.mark.parametrize("mod", ["pyspark", "torch", "numpy", "pyarrow",
                                 "pandas", "tokenizers", "matplotlib"])
def test_imports(mod):
    pytest.importorskip(mod, reason=f"{mod} missing: pip install -r requirements.txt")


@pytest.fixture(scope="module")
def spark():
    """A real local session. Importing pyspark proves nothing on its own."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession

    s = (SparkSession.builder.appName("test_env").master("local[2]")
         .config("spark.ui.enabled", "false")
         .config("spark.ui.showConsoleProgress", "false")
         .config("spark.sql.shuffle.partitions", "4")
         .getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def test_spark_shuffle(spark):
    from pyspark.sql import functions as F

    df = spark.range(0, 10_000).withColumn("g", F.col("id") % 7)
    assert df.groupBy("g").count().count() == 7


def test_spark_python_udf(spark):
    """The check that actually catches broken PySpark installs.

    If the Python interpreter Spark launches for its workers is not the
    one running the driver, the import succeeds, the session starts,
    the shuffle above passes, and only this fails -- with an error that
    does not mention the real cause. Export PYSPARK_PYTHON to fix it.
    """
    from pyspark.sql import functions as F, types as T

    shout = F.udf(lambda s: (s or "").upper(), T.StringType())
    row = spark.createDataFrame([("ok",)], ["s"]).select(shout("s").alias("u")).collect()
    assert row[0]["u"] == "OK", (
        "Spark Python UDFs are broken. Your worker Python probably differs "
        f"from your driver Python. Try: export PYSPARK_PYTHON={sys.executable}"
    )


def test_reads_warc_records(tmp_path, spark):
    """Read WET records the way Part 1 stage 1 will.

    Not a formality. WARC records are separated by a literal `WARC/1.0`
    line, and Spark can use that as a record delimiter -- but the files
    are CRLF, so anything that assumes bare newlines behaves oddly and
    fails silently rather than raising. Get in the habit of asserting on
    row counts you worked out by hand.
    """
    import gzip

    path = tmp_path / "t.warc.wet.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("WARC/1.0\r\nWARC-Type: warcinfo\r\n\r\nmetadata\r\n")
        for i in (1, 2, 3):
            f.write(f"WARC/1.0\r\nWARC-Target-URI: http://x{i}.com/\r\n"
                    f"\r\npage {i} body text\r\n")

    df = spark.read.text(str(path), lineSep="WARC/1.0")
    # Splitting on N delimiters yields N+1 fields: the first is the empty
    # text before the first marker. Stage 1 drops it along with the
    # warcinfo record, because neither has a Target-URI.
    assert df.count() == 5, (
        f"expected 5 fields (empty + warcinfo + 3 pages), got {df.count()}. "
        "Stage 1 uses this delimiter to split WET files.")
