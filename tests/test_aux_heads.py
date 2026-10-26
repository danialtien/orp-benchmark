"""Tests for the IRRA auxiliary objectives grafted onto ``train_lora``.

The masking tests run anywhere (no torch). The parameter-count test needs torch
and is skipped without it -- run it on the GPU box, where the number it asserts
is the one that goes into the paper's train-time column.
"""
import numpy as np
import pytest

from sar_person.encoders import _mask_tokens

VOCAB = 49408
MASK_ID = 49405          # <|mask|>
HI = VOCAB - 3           # maskable range is 1..49404
SOT, EOT = 49406, 49407


def _caption(n_real=30, length=77):
    """A plausible tokenised caption: SOT, real tokens, EOT, then zero padding."""
    toks = [SOT] + list(range(100, 100 + n_real)) + [EOT]
    return toks + [0] * (length - len(toks))


def test_specials_and_padding_are_never_touched():
    rng = np.random.default_rng(0)
    for _ in range(200):
        ids, labels = _mask_tokens(_caption(), MASK_ID, HI, rng)
        assert ids[0] == SOT
        assert labels[0] == 0
        assert ids[31] == EOT and labels[31] == 0
        assert all(v == 0 for v in ids[32:]), "padding was rewritten"
        assert all(v == 0 for v in labels[32:]), "padding got a label"


def test_mask_rate_is_about_15_percent():
    rng = np.random.default_rng(1)
    selected = total = 0
    for _ in range(500):
        _, labels = _mask_tokens(_caption(), MASK_ID, HI, rng)
        selected += sum(1 for v in labels if v)
        total += 30
    rate = selected / total
    assert 0.12 < rate < 0.18, f"masking rate {rate:.3f} is not ~0.15"


def test_labels_carry_the_original_token():
    rng = np.random.default_rng(2)
    src = _caption()
    ids, labels = _mask_tokens(src, MASK_ID, HI, rng)
    assert src == _caption(), "_mask_tokens mutated its input"
    for i, lab in enumerate(labels):
        if lab:
            assert lab == src[i], "label must be the ORIGINAL token, not the mask"


def test_eighty_ten_ten_split():
    """80% -> <|mask|>, 10% -> random token, 10% -> unchanged."""
    rng = np.random.default_rng(3)
    masked = randomised = kept = 0
    for _ in range(800):
        src = _caption()
        ids, labels = _mask_tokens(src, MASK_ID, HI, rng)
        for i, lab in enumerate(labels):
            if not lab:
                continue
            if ids[i] == MASK_ID:
                masked += 1
            elif ids[i] == src[i]:
                kept += 1
            else:
                randomised += 1
    n = masked + randomised + kept
    assert 0.75 < masked / n < 0.85
    assert 0.05 < randomised / n < 0.15
    assert 0.05 < kept / n < 0.15


def test_at_least_one_token_is_always_masked():
    """A caption with nothing selected contributes no gradient at all."""
    rng = np.random.default_rng(4)
    for _ in range(300):
        _, labels = _mask_tokens(_caption(n_real=2), MASK_ID, HI, rng)
        assert any(labels), "every sample must carry at least one MLM label"


def test_masking_is_reproducible_from_the_seed():
    a = _mask_tokens(_caption(), MASK_ID, HI, np.random.default_rng((0, 3, 17)))
    b = _mask_tokens(_caption(), MASK_ID, HI, np.random.default_rng((0, 3, 17)))
    c = _mask_tokens(_caption(), MASK_ID, HI, np.random.default_rng((0, 4, 17)))
    assert a == b, "--deterministic would be a lie"
    assert a != c, "a different epoch must give a different mask"


# --------------------------------------------------------------------------- #
# torch-only                                                                   #
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch", reason="parameter counts need torch")


def _heads(**kw):
    from sar_person.encoders import _aux_heads_cls
    return _aux_heads_cls()(**kw)


def test_id_head_param_count():
    """512 x N_id + N_id. ICFG has 3102 train identities."""
    aux = _heads(embed_dim=512, num_classes=3102, mlm=False)
    assert aux.n_params() == 1_591_326
    aux = _heads(embed_dim=512, num_classes=11003, mlm=False)   # CUHK
    assert aux.n_params() == 5_644_539


def test_mlm_branch_param_count():
    """cross_attn 1_050_624 + cmt(4) 12_609_536 + 3 LN 3_072 + head 25_609_984."""
    aux = _heads(embed_dim=512, num_classes=0, mlm=True, cmt_depth=4,
                 vocab_size=49408)
    assert aux.n_params() == 39_273_216


def test_full_aux_param_count_icfg():
    """The number that goes in the paper's TRAIN-TIME column for ICFG."""
    aux = _heads(embed_dim=512, num_classes=3102, mlm=True, cmt_depth=4,
                 vocab_size=49408)
    assert aux.n_params() == 40_864_542


def test_id_classifier_needs_unnormalised_features():
    """Regression guard for the bug that silently kills the ID loss.

    Feeding L2-normalised features to a linear classifier confines every logit
    to a ball of radius ||W||, so the cross-entropy is nearly flat and the loss
    becomes noise. IRRA (build.py:113) passes the raw pooled feature.
    """
    aux = _heads(embed_dim=512, num_classes=64, mlm=False)
    torch.manual_seed(0)
    f_raw = torch.randn(8, 512) * 6.0                 # CLIP features are not unit
    f_norm = f_raw / f_raw.norm(dim=-1, keepdim=True)
    pid = torch.arange(8)
    ce = torch.nn.functional.cross_entropy
    loss_raw = ce(aux.classifier(f_raw), pid)
    loss_norm = ce(aux.classifier(f_norm), pid)
    spread_raw = aux.classifier(f_raw).std()
    spread_norm = aux.classifier(f_norm).std()
    assert spread_norm < spread_raw / 4, (
        "normalised features should collapse the logit spread -- if this fails "
        "the test's premise is wrong, not the code")
    assert not torch.isclose(loss_raw, loss_norm)


def test_cross_former_shapes():
    aux = _heads(embed_dim=512, num_classes=0, mlm=True, cmt_depth=2)
    q = torch.randn(2, 77, 512)          # masked text tokens
    kv = torch.randn(2, 193, 512)        # image tokens (CLS + patches)
    out = aux.cross_former(q, kv, kv)
    assert out.shape == (2, 77, 512)
    scores = aux.mlm_head(out).reshape(-1, aux.vocab_size)
    assert scores.shape == (2 * 77, 49408)
