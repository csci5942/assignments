"""Build a TinyStories twin of the A1 training data.

    python prepare_tinystories.py

Downloads the head of TinyStories (roneneldan/TinyStories on Hugging
Face, the plain-text train file), keeps enough text for the same 82M-char
budget the A1 base run consumed (5000 iters x 64 batch x 256 block =
81,920,000 chars), and encodes it with the A1 vocabulary under the
lecture's normalization + OOV protocol (protocol.py). Output lands in
data-tinystories/: train.bin / val.bin / meta.json, drop-in compatible
with the A1 train.py.

Two prep decisions, made here so your numbers match everyone's:

- The raw train file separates stories with "<|endoftext|>". That is a
  GPT-2 tokenizer artifact, not text; it becomes a blank line BEFORE the
  protocol runs. (The provided eval corpus keeps the protocol exact and
  therefore contains "endoftext" residue; there it is measurement
  material, here it would be 82M chars of training noise.)
- val.bin is encoded from corpora/tinystories.filtered.txt, byte for
  byte. The twin's val split IS its eval-grid column, exactly as A1's
  val split is the shakespeare column.

A1 trained on a 1.1M-char corpus for about 74 epochs. The twin sees 82M
unique chars about once. Same budget, very different data diversity;
Part 3 asks what that does.
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

import protocol

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStories-train.txt")
A1_BUDGET = 5000 * 64 * 256       # max_iters * batch_size * block_size
CHUNK = 32 * 1024 * 1024          # bytes per download request


def find_a1():
    """The A1 checkout: ../assignment-1 (the course repo name), or
    ../01-shakespeare (the instructor workspace)."""
    for name in ("assignment-1", "01-shakespeare"):
        path = os.path.normpath(os.path.join(HERE, "..", name))
        if os.path.isdir(path):
            return path
    return os.path.normpath(os.path.join(HERE, "..", "assignment-1"))


def clean(raw_bytes, vocab):
    """Raw bytes to filtered training text: decode, unify newlines, turn
    the story separator into a blank line, then the lecture protocol."""
    text = raw_bytes.decode("utf-8", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("<|endoftext|>", "\n\n")
    return protocol.filter_oov(text, vocab)


def download_train_text(cache_path, vocab, target_chars):
    """Fetch the head of the train file until the cleaned text covers the
    budget. The raw bytes are cached, so a rerun needs no network."""
    fetched = os.path.getsize(cache_path) if os.path.exists(cache_path) else 0
    while True:
        if fetched:
            with open(cache_path, "rb") as f:
                text, stats = clean(f.read(), vocab)
            print(f"  cache {fetched / 1e6:.0f} MB -> {len(text):,} filtered chars "
                  f"(oov {stats['oov_fraction_after_normalization']:.2%})")
            if len(text) >= target_chars:
                return text, stats
        # Cleaning keeps about 95% of the bytes; ask for that much extra.
        need = max(int(target_chars / 0.95) + 1 - fetched, CHUNK)
        print(f"  fetching bytes {fetched:,} to {fetched + need:,} of TinyStories-train.txt")
        req = urllib.request.Request(TRAIN_URL, headers={
            "User-Agent": "csci5942-a2-prepare/1.0",
            "Range": f"bytes={fetched}-{fetched + need - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            chunk = r.read()
        if not chunk:
            sys.exit("train file exhausted before the budget was covered; lower --train-chars")
        with open(cache_path, "ab") as f:
            f.write(chunk)
        fetched += len(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1-dir", default=find_a1(),
                    help="A1 checkout, for data/meta.json (the vocabulary)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "data-tinystories"))
    ap.add_argument("--train-chars", type=int, default=A1_BUDGET,
                    help="filtered training chars to keep (default: the A1 budget, 81,920,000)")
    args = ap.parse_args()

    with open(os.path.join(args.a1_dir, "data", "meta.json")) as f:
        meta = json.load(f)
    itos = meta["itos"]
    stoi = {ch: i for i, ch in enumerate(itos)}
    vocab = set(itos)
    os.makedirs(args.out_dir, exist_ok=True)

    print("== val (the twin's held-out split = its eval-grid column) ==")
    val_path = os.path.join(HERE, "corpora", "tinystories.filtered.txt")
    if not os.path.exists(val_path):
        sys.exit(f"missing {val_path}; restore it from the repo, the corpora are fixed")
    with open(val_path, encoding="utf-8") as f:
        val_text = f.read()
    print(f"  {len(val_text):,} chars from corpora/tinystories.filtered.txt")

    print("== train ==")
    cache = os.path.join(args.out_dir, "tinystories-train.head.txt")
    train_text, train_stats = download_train_text(cache, vocab, args.train_chars)
    train_text = train_text[:args.train_chars]

    train_ids = np.array([stoi[c] for c in train_text], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_text], dtype=np.uint16)
    train_ids.tofile(os.path.join(args.out_dir, "train.bin"))
    val_ids.tofile(os.path.join(args.out_dir, "val.bin"))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"vocab_size": meta["vocab_size"], "itos": itos,
                   "source": "roneneldan/TinyStories, A1 vocabulary, lecture normalization "
                             "+ OOV protocol, <|endoftext|> stripped from train",
                   "train_chars": len(train_ids), "val_chars": len(val_ids),
                   "val_source": "corpora/tinystories.filtered.txt (provided)",
                   "oov_fraction_train": train_stats["oov_fraction_after_normalization"]}, f)

    print(f"train {len(train_ids):,} tokens, val {len(val_ids):,} tokens, vocab {meta['vocab_size']}")
    print(f"""
Next (A1's train.py reads the data/ directory next to itself):

    cd {os.path.relpath(args.a1_dir, os.getcwd())}
    mv data data.shakespeare
    ln -s {os.path.abspath(args.out_dir)} data
    python train.py --config configs/base.json --out-dir out/base-tinystories
    rm data && mv data.shakespeare data
""")


if __name__ == "__main__":
    main()
