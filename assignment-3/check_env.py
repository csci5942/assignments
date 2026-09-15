#!/usr/bin/env python3
"""Environment check for CSCI 5942 Assignment 3.

    python check_env.py

Prints a pass/fail line per requirement and, for anything that fails,
what to do about it. Exits nonzero if a required check fails, so it
can gate a setup script. Nothing here touches the network except the
corpus reachability check, which is advisory.
"""

import os
import platform
import re
import shutil
import subprocess
import sys

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "
_failures: list[str] = []
_warnings: list[str] = []


def report(status: str, name: str, detail: str = "", fix: str = "") -> None:
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if status == BAD:
        _failures.append(f"{name}: {fix or detail}")
    elif status == WARN:
        _warnings.append(f"{name}: {fix or detail}")


# --------------------------------------------------------------------
# 1. Python
# --------------------------------------------------------------------
def check_python() -> None:
    v = sys.version_info
    got = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 10):
        report(OK, "Python 3.10+", got)
    else:
        report(BAD, "Python 3.10+", f"found {got}",
               "install Python 3.10 or newer and rebuild your venv")

    if sys.prefix != sys.base_prefix or os.environ.get("VIRTUAL_ENV"):
        report(OK, "virtualenv active", os.path.basename(sys.prefix))
    else:
        report(WARN, "virtualenv active", "not in a venv",
               "python -m venv .venv && source .venv/bin/activate")


# --------------------------------------------------------------------
# 2. Java (PySpark needs a JVM; pip cannot provide one)
# --------------------------------------------------------------------
JAVA_HELP = {
    "Linux": "sudo apt install openjdk-17-jre-headless",
    "Darwin": "brew install openjdk@17   "
              "(then follow brew's caveat to symlink it)",
    "Windows": "install Temurin 17 from adoptium.net, "
               "or use WSL2 and follow the Linux instructions",
}

HPC_HELP = ("this looks like an HPC login node, where you cannot install "
            "packages yourself. Find the JVM module instead:\n"
            "         module spider java     (or: module avail | grep -i jdk)\n"
            "       then `module load` whatever it reports, and re-run this check")


def on_hpc() -> bool:
    """Detect an Lmod/environment-modules cluster (Delta, Chameleon bare metal).

    Matters because the apt/brew advice below is not just unhelpful there,
    it is wrong: students have no sudo and get their JVM from `module load`.
    """
    return any(os.environ.get(v) for v in
               ("LMOD_CMD", "MODULESHOME", "MODULEPATH", "SLURM_CLUSTER_NAME"))


def check_java() -> None:
    hint = (HPC_HELP if on_hpc()
            else JAVA_HELP.get(platform.system(), "install a Java 17+ runtime"))
    exe = shutil.which("java")
    if not exe:
        report(BAD, "Java 17+ runtime", "java not on PATH", hint)
        return
    try:
        out = subprocess.run(["java", "-version"], capture_output=True,
                             text=True, timeout=30)
        text = (out.stderr or "") + (out.stdout or "")
        m = re.search(r'version "?(\d+)', text)
        major = int(m.group(1)) if m else 0
    except Exception as e:  # noqa: BLE001
        report(BAD, "Java 17+ runtime", f"could not run java ({e})", hint)
        return

    if major >= 17:
        report(OK, "Java 17+ runtime", f"java {major} at {exe}")
    else:
        report(BAD, "Java 17+ runtime", f"found java {major}", hint)

    if os.environ.get("JAVA_HOME"):
        report(OK, "JAVA_HOME set", os.environ["JAVA_HOME"])
    else:
        report(WARN, "JAVA_HOME set", "unset",
               "usually fine, but set it if Spark fails to start")


# --------------------------------------------------------------------
# 3. Packages
# --------------------------------------------------------------------
def check_imports() -> None:
    for mod, label in [("pyspark", "pyspark"), ("torch", "torch"),
                       ("numpy", "numpy"), ("pyarrow", "pyarrow"),
                       ("pandas", "pandas"), ("tokenizers", "tokenizers"),
                       ("matplotlib", "matplotlib"), ("pytest", "pytest")]:
        try:
            m = __import__(mod)
            report(OK, f"import {label}", getattr(m, "__version__", "?"))
        except Exception as e:  # noqa: BLE001
            report(BAD, f"import {label}", str(e),
                   "pip install -r requirements.txt")


# --------------------------------------------------------------------
# 4. Spark actually runs a job, including a Python UDF
# --------------------------------------------------------------------
def check_spark() -> None:
    """Importing pyspark proves nothing. This starts a real session,
    runs a shuffle, and round-trips a Python UDF.

    The UDF is the part that matters. If the Python interpreter Spark
    launches for its workers is not the one running the driver, the
    import and the session both succeed and only the UDF fails, with
    an error that does not mention the real cause.
    """
    try:
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        from pyspark.sql import SparkSession, functions as F, types as T

        spark = (SparkSession.builder.appName("envcheck").master("local[2]")
                 .config("spark.ui.enabled", "false")
                 .config("spark.ui.showConsoleProgress", "false")
                 .config("spark.sql.shuffle.partitions", "4")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        report(OK, "Spark session starts", f"spark {spark.version}")

        df = spark.range(0, 10_000).withColumn("g", F.col("id") % 7)
        n = df.groupBy("g").count().count()
        assert n == 7, f"shuffle produced {n} groups, expected 7"
        report(OK, "Spark shuffle works", "7 groups")

        shout = F.udf(lambda s: (s or "").upper(), T.StringType())
        got = (spark.createDataFrame([("ok",)], ["s"])
               .select(shout("s").alias("u")).collect()[0]["u"])
        assert got == "OK", f"UDF returned {got!r}"
        report(OK, "Spark Python UDF works", "driver/worker python match")

        spark.stop()
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0][:160]
        report(BAD, "Spark smoke test", msg,
               "if the session started but the UDF failed, your worker "
               "Python differs from your driver Python; export "
               f"PYSPARK_PYTHON={sys.executable}")


# --------------------------------------------------------------------
# 5. Torch device (informational: Parts 1-2 are CPU-only)
# --------------------------------------------------------------------
def check_torch_device() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            report(OK, "GPU visible to torch", torch.cuda.get_device_name(0))
        else:
            report(WARN, "GPU visible to torch", "cpu only",
                   "fine for Parts 1-2; Parts 3-4 need a GPU")
    except Exception as e:  # noqa: BLE001
        report(WARN, "GPU visible to torch", str(e)[:80])


# --------------------------------------------------------------------
# 6. Machine capacity (the README runtime table assumes 8 cores)
# --------------------------------------------------------------------
def check_capacity() -> None:
    cores = os.cpu_count() or 1
    detail = f"{cores} cores"
    if cores >= 8:
        report(OK, "CPU cores", detail)
    elif cores >= 4:
        report(WARN, "CPU cores", detail,
               "README timings assume 8; expect roughly 2x longer")
    else:
        report(WARN, "CPU cores", detail,
               "Part 1 will be slow here; use a cloud VM for the graded run")

    try:
        free_gb = shutil.disk_usage(os.path.dirname(os.path.abspath(__file__))).free / 1e9
        if free_gb >= 15:
            report(OK, "free disk", f"{free_gb:.0f} GB")
        else:
            report(WARN, "free disk", f"{free_gb:.0f} GB",
                   "the 2GB corpus plus Parquet output and shards wants ~15GB")
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------
# 7. Corpus reachability (advisory)
# --------------------------------------------------------------------
def check_corpus() -> None:
    url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-33/wet.paths.gz"
    try:
        import requests
        r = requests.get(url, timeout=20)
        if r.ok:
            report(OK, "corpus reachable", "Common Crawl responds")
        else:
            report(WARN, "corpus reachable", f"HTTP {r.status_code}",
                   "check your network; you can fetch the corpus later")
    except Exception as e:  # noqa: BLE001
        report(WARN, "corpus reachable", str(e)[:80],
               "check your network; you can fetch the corpus later")


def main() -> int:
    print(f"CSCI 5942 Assignment 3 environment check")
    print(f"{platform.platform()}\n")
    check_python()
    check_java()
    check_imports()
    check_spark()
    check_torch_device()
    check_capacity()
    check_corpus()

    print()
    if _failures:
        print(f"{len(_failures)} required check(s) failed:\n")
        for f in _failures:
            print(f"  - {f}")
        print("\nFix these before starting Part 1.")
        return 1

    if _warnings:
        print(f"All required checks passed, with {len(_warnings)} warning(s):\n")
        for w in _warnings:
            print(f"  - {w}")
        print()
    print("Environment is ready. Next: python data/fetch.py --files 1 --out data/dev/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
