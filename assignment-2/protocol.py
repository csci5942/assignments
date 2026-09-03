"""The evaluation protocol from Lecture 4, shared by every script here.

Two things live in this file: the normalization + OOV-deletion pipeline
(how much of a foreign corpus the 65-char Shakespeare vocabulary can
even represent), and the teacher-forced loss measurement. This is the
exact code the lecture figures were computed with. Do not change it;
if you do, your numbers stop being comparable to anyone else's.

Protocol:
  1. Normalize: curly quotes to straight quotes, unicode dashes to "-",
     ellipsis to "...", unicode spaces to " ", then NFKD diacritic
     stripping for any remaining char whose stripped base is in-vocab.
  2. Measure oov_fraction over the normalized text.
  3. Delete every remaining OOV char. The model never sees them.
  4. Teacher-forced mean cross-entropy in nats/char, block_size 256,
     non-overlapping full blocks, at most MAX_BLOCKS blocks per corpus,
     eval mode (dropout off), seed 1337.
"""

import unicodedata

import numpy as np
import torch

BLOCK = 256        # matches every A1 config's block_size
MAX_BLOCKS = 400   # cap per corpus, so every column costs the same
BATCH = 64         # eval batch, memory only, no effect on the number
SEED = 1337

NORM_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", "　": " ",
    "​": "", "﻿": "",
}


def normalize(text, vocab):
    """Apply the normalization map, then NFKD diacritic stripping for any
    remaining out-of-vocab char whose combining-mark-stripped base is fully
    in-vocab. OOV chars are left in place for the measurement step."""
    out = []
    for ch in text:
        if ch in NORM_MAP:
            out.append(NORM_MAP[ch])
        elif ch in vocab:
            out.append(ch)
        else:
            base = "".join(c for c in unicodedata.normalize("NFKD", ch)
                           if not unicodedata.combining(c))
            if base and all(c in vocab for c in base):
                out.append(base)
            else:
                out.append(ch)
    return "".join(out)


def filter_oov(raw, vocab):
    """Normalize, measure what the vocabulary cannot represent, then
    delete it. Returns (filtered_text, stats)."""
    norm = normalize(raw, vocab)
    kept = [ch for ch in norm if ch in vocab]
    oov = len(norm) - len(kept)
    dropped = {}
    for ch in norm:
        if ch not in vocab:
            dropped[ch] = dropped.get(ch, 0) + 1
    top_dropped = sorted(dropped.items(), key=lambda kv: -kv[1])[:12]
    stats = {
        "raw_chars": len(raw),
        "normalized_chars": len(norm),
        "oov_chars": oov,
        "oov_fraction_after_normalization": round(oov / max(len(norm), 1), 6),
        "retained_chars": len(kept),
        "top_deleted_chars": [[repr(c)[1:-1], n] for c, n in top_dropped],
    }
    return "".join(kept), stats


@torch.no_grad()
def nats_per_char(model, text, stoi, device="cpu"):
    """Mean teacher-forced cross-entropy in nats/char over non-overlapping
    full blocks of BLOCK chars, capped at MAX_BLOCKS blocks. The text must
    already be filtered (every char in vocab)."""
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    n_blocks = min((len(ids) - 1) // BLOCK, MAX_BLOCKS)
    if n_blocks == 0:
        raise ValueError(f"corpus too short: {len(text)} chars < {BLOCK + 1}")
    x = torch.from_numpy(ids[:n_blocks * BLOCK].reshape(n_blocks, BLOCK))
    y = torch.from_numpy(ids[1:n_blocks * BLOCK + 1].reshape(n_blocks, BLOCK))
    total, count = 0.0, 0
    for i in range(0, n_blocks, BATCH):
        xb = x[i:i + BATCH].to(device)
        yb = y[i:i + BATCH].to(device)
        _, loss = model(xb, yb)
        total += loss.item() * xb.numel()
        count += xb.numel()
    return total / count, n_blocks
