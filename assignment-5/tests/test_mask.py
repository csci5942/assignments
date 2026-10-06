"""Part 2: the template and the loss mask (CPU, no model)."""
import sft


def test_target_starts_with_space_and_ends_with_eot():
    p, t = sft.format_example("name[The Vaults], eatType[pub]", "The Vaults is a pub.")
    assert t[-1] == sft.EOT
    assert sft.ENC.decode(t[:-1]) == " The Vaults is a pub."
    assert sft.ENC.decode(p).endswith("Description:")


def test_mask_scores_only_the_target():
    p, t = sft.format_example("name[Alimentum], area[city centre]", "Alimentum is in the city centre.")
    x, y = sft.collate([(p, t)])
    assert x.shape == y.shape == (1, len(p) + len(t) - 1)
    # x is the sequence without its last token; y is x shifted left by one
    assert x[0].tolist() == (p + t)[:-1]
    assert y[0, :len(p) - 1].tolist() == [sft.IGNORE] * (len(p) - 1), "prompt positions must be ignored"
    assert y[0, len(p) - 1:].tolist() == t, "every target token, including EOT, must be scored"
    assert (y[0] != sft.IGNORE).sum().item() == len(t)


def test_padding_is_ignored_and_right_aligned():
    a = sft.format_example("name[Aromi], food[Chinese]", "Aromi serves Chinese food.")
    b = sft.format_example("name[Bibimbap House]", "Bibimbap House is a restaurant in the riverside area with a high customer rating.")
    x, y = sft.collate([a, b])
    T = len(b[0]) + len(b[1]) - 1
    assert x.shape == (2, T)
    la = len(a[0]) + len(a[1]) - 1
    assert (x[0, la:] == sft.EOT).all(), "pad with EOT"
    assert (y[0, la:] == sft.IGNORE).all(), "padding carries no loss"
    assert (y[1] != sft.IGNORE).sum().item() == len(b[1])
