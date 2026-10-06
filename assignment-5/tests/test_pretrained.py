"""Part 0: the published GPT-2 weights load into A4's model class.
Downloads gpt2 (about 550 MB) on first run; the loss is pinned from a run
of the instructor's reference (reference/parity_check.py)."""
import torch

import sft
from model import from_pretrained

PARA = ("The Eagle is a coffee shop near Burger King in the city centre. It serves Italian food "
        "at a moderate price and has a customer rating of 3 out of 5. It is not family friendly.")
EXPECTED_LOSS = 3.3211   # from reference/parity_check.py, 2026-10-02 (HF: 3.3217)


def test_gpt2_small_loads_and_reads_english():
    model = from_pretrained("gpt2").eval()
    assert model.num_params() == 163_037_184       # 124.4M tied, plus the untied 38.6M head
    ids = torch.tensor([sft.ENC.encode(PARA)])
    with torch.no_grad():
        _, loss = model(ids[:, :-1], ids[:, 1:])
    assert loss.item() < 4.0, "a loaded GPT-2 scores plain English well under 4 nats per token"
    if EXPECTED_LOSS is not None:
        assert abs(loss.item() - EXPECTED_LOSS) < 0.05
