#!/usr/bin/env python3
"""Fetch Common Crawl WET files.

    python data/fetch.py --files 1 --out data/dev/    # ~80MB, development
    python data/fetch.py --files 5 --out data/raw/    # ~380MB, graded set

A WET file is plain text already extracted from HTML by Common Crawl:
one WARC record per page, carrying the source URL and the page's visible
text. Roughly 80MB gzipped each, about 300MB of text.

Which files you get is deterministic -- the first N of a fixed crawl's
path listing -- so everyone's corpus is byte-identical and the funnel
numbers are comparable. `--crawl` pins the crawl; leave it alone unless
you have a reason.

Re-running is safe: a file already present with the right size is
skipped, so an interrupted download resumes.
"""

import argparse
import gzip
import io
import os
import sys

import requests
from tqdm import tqdm

BASE = "https://data.commoncrawl.org"
DEFAULT_CRAWL = "CC-MAIN-2024-33"
RETRIES = 4
BACKOFF = 1.5


class FetchError(RuntimeError):
    """A request that failed in a way worth reporting."""


def _get(session, url, stream=False):
    """GET with retries on transient failures only. A 404 is not
    transient, so retrying it just makes a wrong crawl name take five
    times as long to tell you it is wrong."""
    import time
    last = None
    for attempt in range(RETRIES):
        try:
            r = session.get(url, timeout=120, stream=stream)
            if r.status_code == 200:
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                raise FetchError(f"HTTP {r.status_code} for {url}")
            last = f"HTTP {r.status_code}"
        except FetchError:
            raise
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(BACKOFF ** attempt)
    raise FetchError(f"GET {url} failed after {RETRIES} tries: {last}")


def wet_paths(crawl, session, n):
    """The first n WET file paths of a crawl, from its path listing."""
    url = f"{BASE}/crawl-data/{crawl}/wet.paths.gz"
    body = _get(session, url).content
    with gzip.open(io.BytesIO(body), "rt") as f:
        paths = [line.strip() for line in f if line.strip()]
    if n > len(paths):
        raise FetchError(f"{crawl} has only {len(paths):,} WET files")
    return paths[:n]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--files", type=int, default=1,
                    help="how many WET files to fetch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crawl", default=DEFAULT_CRAWL)
    args = ap.parse_args()

    if args.files < 1:
        ap.error("--files must be at least 1")
    os.makedirs(args.out, exist_ok=True)
    session = requests.Session()

    paths = wet_paths(args.crawl, session, args.files)
    print(f"{args.crawl}: fetching {len(paths)} WET file(s) -> {args.out}")

    total = 0
    for i, p in enumerate(paths, 1):
        dest = os.path.join(args.out, f"cc{i:03d}.warc.wet.gz")
        url = f"{BASE}/{p}"
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"  [{i}/{len(paths)}] {os.path.basename(dest)} already present")
            total += os.path.getsize(dest)
            continue
        r = _get(session, url, stream=True)
        size = int(r.headers.get("content-length", 0))
        tmp = dest + ".part"
        with open(tmp, "wb") as fh, tqdm(
                total=size or None, unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"[{i}/{len(paths)}] {os.path.basename(dest)}") as bar:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
        os.replace(tmp, dest)      # atomic: an interrupt cannot leave a
        total += os.path.getsize(dest)   # truncated file looking complete

    print(f"\n{len(paths)} file(s), {total/1e6:.0f} MB in {args.out}")
    print("next: python pipeline/build_corpus.py "
          f"--input '{os.path.join(args.out, '*.warc.wet.gz')}' --out out/dev")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FetchError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        print("Check --crawl and your network. Crawl names look like "
              "CC-MAIN-2024-33; see https://commoncrawl.org/get-started",
              file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrupted. Re-run to resume; finished files are skipped.",
              file=sys.stderr)
        sys.exit(130)
