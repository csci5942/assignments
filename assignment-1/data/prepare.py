"""Download tiny-shakespeare and encode it at the character level.

Produces train.bin / val.bin (uint16 token ids) and meta.json with the
vocabulary. Run once:

    python data/prepare.py
"""

import json
import os
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    txt_path = os.path.join(HERE, "input.txt")
    if not os.path.exists(txt_path):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        data = f.read()
    print(f"{len(data):,} characters")

    chars = sorted(set(data))
    stoi = {ch: i for i, ch in enumerate(chars)}
    print(f"vocab size {len(chars)}")

    n = len(data)
    train_ids = np.array([stoi[c] for c in data[: int(n * 0.9)]], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in data[int(n * 0.9):]], dtype=np.uint16)
    train_ids.tofile(os.path.join(HERE, "train.bin"))
    val_ids.tofile(os.path.join(HERE, "val.bin"))

    with open(os.path.join(HERE, "meta.json"), "w") as f:
        json.dump({"vocab_size": len(chars), "itos": chars}, f)
    print(f"train {len(train_ids):,} tokens, val {len(val_ids):,} tokens")


if __name__ == "__main__":
    main()
