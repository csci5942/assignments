"""SFT data: the template and the loss mask. Part 2 edits `format_example`
and `collate`; the rest is given.

An E2E example is a meaning representation (prompt) and a reference
(target). For training it becomes one sequence,

    <prompt tokens> <target tokens> <|endoftext|>

with loss on the target span and its end-of-text token only (lecture 12,
"Templates and Masking"). Tokenizer: A4's GPT-2 BPE through tiktoken.
"""

import json
import os

import tiktoken
import torch

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token                 # 50256, <|endoftext|>
IGNORE = -100                       # F.cross_entropy ignores targets of -100

# The template. At generation time the prompt ends after the colon.
PROMPT = "{mr}\nDescription:"

HERE = os.path.dirname(os.path.abspath(__file__))
E2E_DIR = os.path.join(HERE, "data", "e2e")


def format_example(mr: str, ref: str) -> tuple[list[int], list[int]]:
    """Return (prompt_ids, target_ids): PROMPT filled with `mr`, and a
    leading space plus `ref` followed by EOT."""
    # --- YOUR IMPLEMENTATION HERE ---
    raise NotImplementedError("implement this block")


def collate(examples: list[tuple[list[int], list[int]]]):
    """Batch formatted examples into (x, y) int64 tensors of shape (B, T):
    y[t] is the token after x[t], as in A1's train.py. Right-pad with EOT
    to the longest row; y is IGNORE on prompt and padding positions and
    the target span (including EOT) is scored."""
    # --- YOUR IMPLEMENTATION HERE ---
    raise NotImplementedError("implement this block")


def load_split(split: str, e2e_dir: str = E2E_DIR) -> list[dict]:
    """Rows of data/e2e/<split>.jsonl, written by data/prepare_e2e.py."""
    path = os.path.join(e2e_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run python data/prepare_e2e.py")
    with open(path) as f:
        return [json.loads(line) for line in f]


def describe(prompt_ids: list[int], target_ids: list[int]) -> str:
    """One formatted example token by token, with the mask beside it."""
    x, y = collate([(prompt_ids, target_ids)])
    toks = [ENC.decode([t]) for t in x[0].tolist()]
    out = []
    for tok, tgt in zip(toks, y[0].tolist()):
        shown = repr(tok)[1:-1]
        out.append(f"{shown:>14s}  {'scored' if tgt != IGNORE else '-'}")
    return "\n".join(out)
