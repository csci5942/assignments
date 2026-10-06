"""Download the E2E NLG Challenge data and write it as jsonl.

    python data/prepare_e2e.py

E2E (Novikova et al. 2017) is the task of the LoRA paper's GPT-2
experiments: a meaning representation of three to eight attributes and a
human-written description. Original release from
github.com/tuetschek/e2e-dataset (CC BY-SA 4.0), original splits:

    train  42,061 pairs over 4,862 meaning representations
    dev     4,672 pairs over   547
    test    4,693 pairs over   630

The test set is written one row per meaning representation with all of
its references (about eight), for multi-reference BLEU in eval.py.

Output: data/e2e/{train,dev}.jsonl with {"mr", "ref"} rows, test.jsonl
with {"mr", "refs"} rows, meta.json with the counts. Pattern: A2's
prepare_tinystories.py. Not edited in Assignment 5.
"""

import csv
import io
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e2e")
BASE = "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/"
FILES = {"train": "trainset.csv", "dev": "devset.csv", "test": "testset_w_refs.csv"}


def fetch(name: str) -> list[dict]:
    raw = urllib.request.urlopen(BASE + name, timeout=120).read().decode("utf8")
    return list(csv.DictReader(io.StringIO(raw)))


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {"source": BASE, "license": "CC BY-SA 4.0", "splits": {}}
    for split, fname in FILES.items():
        rows = fetch(fname)
        path = os.path.join(OUT, f"{split}.jsonl")
        if split == "test":
            grouped: dict[str, list[str]] = {}
            for r in rows:
                grouped.setdefault(r["mr"], []).append(r["ref"].strip())
            with open(path, "w") as f:
                for mr, refs in grouped.items():
                    f.write(json.dumps({"mr": mr, "refs": refs}) + "\n")
            meta["splits"][split] = {"pairs": len(rows), "mrs": len(grouped)}
        else:
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps({"mr": r["mr"], "ref": r["ref"].strip()}) + "\n")
            meta["splits"][split] = {"pairs": len(rows),
                                     "mrs": len({r["mr"] for r in rows})}
        print(f"{split:5s} {meta['splits'][split]['pairs']:>6,} pairs  "
              f"{meta['splits'][split]['mrs']:>5,} meaning representations  -> {path}")
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
