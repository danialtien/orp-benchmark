"""Tests for the train-time image augmentation (C1 in RECIPE_R_STAR.md).

The one bug that would poison every number in the paper is augmentation leaking
into the evaluation path, so that is what most of these check.
"""
import numpy as np
import pytest

PIL = pytest.importorskip("PIL", reason="augmentation needs Pillow")
from PIL import Image                                            # noqa: E402

from sar_person.encoders import _geom_aug, _random_erase         # noqa: E402


def _img(w=128, h=384):
    a = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    return Image.fromarray(a)


def test_geom_aug_preserves_size():
    rng = np.random.default_rng(0)
    for _ in range(50):
        out = _geom_aug(_img(), rng)
        assert out.size == (128, 384), "crop must return the input resolution"


def test_geom_aug_flips_about_half_the_time():
    rng = np.random.default_rng(1)
    src = _img()
    flipped = 0
    for _ in range(400):
        out = _geom_aug(src, rng, pad=0)          # pad 0 isolates the flip
        if np.array_equal(np.asarray(out), np.asarray(src)[:, ::-1]):
            flipped += 1
    assert 0.4 < flipped / 400 < 0.6


def test_geom_aug_is_reproducible():
    a = _geom_aug(_img(), np.random.default_rng((0, 2, 5)))
    b = _geom_aug(_img(), np.random.default_rng((0, 2, 5)))
    c = _geom_aug(_img(), np.random.default_rng((0, 3, 5)))
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))


def test_random_erase_fires_about_half_the_time_and_zeroes_a_patch():
    rng = np.random.default_rng(2)
    fired = 0
    for _ in range(400):
        a = np.ones((384, 128, 3), np.float32)
        out = _random_erase(a, rng)
        if out is not a and (out == 0.0).any():
            fired += 1
            frac = float((out == 0.0).all(axis=2).mean())
            assert 0.01 < frac < 0.45, f"erased fraction {frac:.3f} out of range"
    assert 0.4 < fired / 400 < 0.6


def test_random_erase_does_not_mutate_its_input():
    rng = np.random.default_rng(3)
    for _ in range(50):
        a = np.ones((384, 128, 3), np.float32)
        _random_erase(a, rng)
        assert (a == 1.0).all(), "erasing wrote through to the caller's array"


def test_eval_path_is_never_augmented():
    """train_rng=None must reproduce the pre-augmentation behaviour exactly.

    ``encode_images`` calls ``_prep_image`` without train_rng. If that ever
    changes, the gallery becomes random and every reported number is noise.
    """
    import inspect
    from sar_person import encoders
    sig = inspect.signature(encoders.ClipEncoder._prep_image)
    assert sig.parameters["train_rng"].default is None
    src = inspect.getsource(encoders.ClipEncoder.encode_images)
    assert "train_rng" not in src, "encode_images must not pass train_rng"
    src = inspect.getsource(encoders.ClipEncoder._prep_image)
    assert src.count("if train_rng is not None:") == 2, (
        "both the geometric and the erasing hook must be guarded")
