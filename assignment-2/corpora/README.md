# The four eval corpora

These are the exact files behind the Lecture 4 cross-domain numbers,
copied from the lecture figure pipeline. Do not edit or regenerate
them; the point of a fixed exam is that everyone takes the same one.

Each file is raw source text passed through the lecture protocol
(`../protocol.py`): normalize quotes/dashes/spaces, strip diacritics
where the base char is in the A1 vocabulary, then delete every
remaining out-of-vocab character. The deleted fraction is part of the
result, not a footnote: it is how much of the corpus the 65-char
Shakespeare vocabulary could not represent at all.

| file | source | OOV deleted |
| :--- | :--- | ---: |
| shakespeare.filtered.txt | A1 `data/val.bin` decoded (the held-out split) | 0.00% |
| tinystories.filtered.txt | roneneldan/TinyStories `TinyStories-valid.txt`, first 500 KB | 0.98% |
| wikipedia.filtered.txt | en.wikipedia.org plain-text article extracts, mixed topics, capped at 500 KB | 2.30% |
| python-code.filtered.txt | Python stdlib source files, concatenated to 500 KB | 7.69% |

Notes:

- The TinyStories file contains the string `endoftext` here and there.
  The raw file separates stories with `<|endoftext|>`; the protocol
  deletes `<`, `|`, `>` as OOV and keeps the in-vocab letters. This is
  the protocol applied exactly, and it stays.
- python-code loses underscores, parentheses, equals signs, and digits.
  What remains is code-shaped prose, not runnable code. Keep that in
  mind when you interpret the code column of your grid.
- shakespeare.filtered.txt is byte-identical to your A1 val split, so
  your base model's number on it should land within a few hundredths
  of your A1 val loss (training eval used random crops; this protocol
  uses non-overlapping blocks).
