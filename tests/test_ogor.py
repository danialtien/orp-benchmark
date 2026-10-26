"""OGor + selection + Rsum. Every expected value is computed by hand below.

The point of these tests is not that the code runs -- it is that the occluded
gallery is the *same* gallery on every machine and every re-run, because that is
the only thing that makes "IRRA vs MGCC vs ours" a comparison rather than three
unrelated numbers.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from sar_person import ogor
from sar_person.eval_protocol import rank_metrics


# --------------------------------------------------------------------------- #
# A synthetic OIL: 15 classes x 4 samples of a half-transparent square         #
# --------------------------------------------------------------------------- #
@pytest.fixture
def oil_root(tmp_path):
    root = tmp_path / "OIL"
    root.mkdir()
    for cls in ogor.OCCLUSION_CLASSES:
        for i in range(1, ogor.SAMPLES_PER_CLASS + 1):
            # 20x10 so w_h_ratio = 2.0 exactly -- makes the resize arithmetic
            # checkable by hand below
            im = Image.new("RGBA", (20, 10), (255, 0, 0, 255))
            im.putpixel((0, 0), (0, 0, 0, 0))          # -> _has_alpha is True
            im.save(root / f"{cls}{i}.png")
    return root


@pytest.fixture
def oil(oil_root):
    return ogor.OIL(oil_root)


def test_library_is_complete(oil):
    assert len(oil) == 60                      # the paper's "60 common samples"
    assert len(oil.classes) == 15


def test_incomplete_library_is_refused(oil_root):
    (oil_root / "car3.png").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        ogor.OIL(oil_root)
    ogor.OIL(oil_root, strict=False)           # opt-out exists, and is explicit


def test_every_class_has_a_ratio_and_an_anchor():
    # a class present in occlusion_list but missing from either dict would
    # KeyError only for the unlucky seed that draws it -- after 20k images
    assert set(ogor.AREA_RATIO) == set(ogor.OCCLUSION_CLASSES)
    assert set(ogor.ANCHOR) == set(ogor.OCCLUSION_CLASSES)
    assert set().union(*ogor.POSITION_SETS.values()) == set(ogor.OCCLUSION_CLASSES)


# --------------------------------------------------------------------------- #
# Geometry -- reproduced from MGCC's Occlusion_Adding.__call__                 #
# --------------------------------------------------------------------------- #
def test_resize_matches_mgcc_formula(oil):
    """MGCC: h = round(sqrt(area/wh)), w = round(sqrt(area*wh)).

    Crop 128x256 -> area 32768. Instance 20x10 -> wh = 2.0.
    For 'chair' (ratio 0.65): target = 21299.2
        h = round(sqrt(21299.2/2)) = round(103.20) = 103
        w = round(sqrt(21299.2*2)) = round(206.39) = 206
    206 > 128, i.e. MGCC lets the occluder overflow the crop horizontally --
    it is clipped by paste(), not rescaled. Asserting on the *painted* result
    would hide that; assert on the arithmetic instead.
    """
    W, H, wh = 128, 256, 2.0
    area = ogor.AREA_RATIO["chair"] * W * H
    assert round(math.sqrt(area / wh)) == 103
    assert round(math.sqrt(area * wh)) == 206


@pytest.mark.parametrize("anchor,expect", [
    ("left_down",    (0, 200)),
    ("right_down",   (60, 200)),
    ("left_up",      (0, 0)),
    ("left_middle",  (0, 72)),     # H//2 - h = 128 - 56
    ("right_middle", (60, 72)),
])
def test_anchor_positions(anchor, expect):
    # W=100 H=256, instance 40x56
    assert ogor.anchor_xy(anchor, 100, 256, 40, 56) == expect


def test_up_classes_land_at_the_top_and_bottom_classes_at_the_bottom(oil):
    """The position prior is the whole idea: an umbrella above the head, a car
    across the shins. If this inverts, the occlusion is nonsense and no metric
    downstream will tell you."""
    img = Image.new("RGB", (128, 256), (255, 255, 255))
    for cls in ("umbrella", "kite"):
        out = ogor.apply(img, oil, seed=0, classes=[cls])
        a = np.asarray(out)
        assert (a[0] != 255).any(), f"{cls} did not touch the top row"
    for cls in ("car", "bench", "bike"):
        out = ogor.apply(img, oil, seed=0, classes=[cls])
        a = np.asarray(out)
        assert (a[-1] != 255).any(), f"{cls} did not touch the bottom row"


def test_apply_is_deterministic_in_the_seed(oil):
    img = Image.new("RGB", (128, 256), (255, 255, 255))
    a = np.asarray(ogor.apply(img, oil, seed=12345))
    for _ in range(4):
        assert np.array_equal(a, np.asarray(ogor.apply(img, oil, seed=12345)))
    # ...and different seeds do something different, or "deterministic" would
    # be trivially satisfied by a no-op
    assert not np.array_equal(a, np.asarray(ogor.apply(img, oil, seed=999)))


def test_level_overrides_every_class_ratio(oil):
    """With level set, coverage must be monotone in level -- that is what makes
    the Tier-2 x-axis readable."""
    img = Image.new("RGB", (128, 256), (255, 255, 255))
    covered = []
    for lv in (0.1, 0.3, 0.6):
        a = np.asarray(ogor.apply(img, oil, seed=7, level=lv, classes=["stone"]))
        covered.append(float((a != 255).any(axis=2).mean()))
    assert covered[0] < covered[1] < covered[2]


def test_jitter_moves_horizontally_only(oil):
    img = Image.new("RGB", (200, 256), (255, 255, 255))
    base = np.asarray(ogor.apply(img, oil, seed=3, classes=["stone"]))
    rows_base = set(np.flatnonzero((base != 255).any(axis=(1, 2))))
    moved = False
    for s in range(20):
        j = np.asarray(ogor.apply(img, oil, seed=s, jitter=0.5, classes=["stone"]))
        rows = set(np.flatnonzero((j != 255).any(axis=(1, 2))))
        assert rows == rows_base            # vertical prior preserved
        moved |= not np.array_equal(j, base)
    assert moved, "jitter never displaced anything"


# --------------------------------------------------------------------------- #
# Selection                                                                    #
# --------------------------------------------------------------------------- #
def test_one_per_id_matches_mgcc_released_code():
    """MGCC: `if last_id != current_id: occlude else: copy`. First image of each
    identity, in gallery order -- NOT 30%, whatever the paper says."""
    pids = ["a", "a", "a", "b", "b", "c"]
    assert ogor.select(pids, "one_per_id") == [True, False, False,
                                               True, False, True]


def test_ratio_is_stratified_per_identity():
    """Sampling 30% globally would occlude every image of some identities and
    none of others. Per-identity keeps every identity's gallery mixed, which is
    the setting both the paper and the code describe."""
    pids = ["a"] * 10 + ["b"] * 10
    mask = ogor.select(pids, "ratio", ratio=0.3, seed=0)
    assert sum(mask[:10]) == 3
    assert sum(mask[10:]) == 3


def test_ratio_never_silently_skips_a_small_identity():
    # round(0.3 * 2) = 1, but round(0.3 * 1) = 0 -> would leave singletons clean
    assert sum(ogor.select(["x"], "ratio", ratio=0.3)) == 1


def test_ratio_selection_survives_reordering():
    """Seeded per pid, not per position: a gallery built in a different order
    must occlude the same *images*, or two runs are not comparable."""
    pids = ["a"] * 6 + ["b"] * 6
    m1 = ogor.select(pids, "ratio", ratio=0.5, seed=1)
    chosen_a = {i for i in range(6) if m1[i]}
    m2 = ogor.select(["a"] * 6, "ratio", ratio=0.5, seed=1)
    assert {i for i in range(6) if m2[i]} == chosen_a


def test_select_all_and_bad_mode():
    assert ogor.select(["a", "b"], "all") == [True, True]
    with pytest.raises(ValueError, match="unknown selection mode"):
        ogor.select(["a"], "nope")


# --------------------------------------------------------------------------- #
# Rsum                                                                         #
# --------------------------------------------------------------------------- #
def test_rsum_is_the_sum_of_the_three_recalls():
    """3 queries, 4 gallery images, pids by hand:

      gallery pids : [x, y, y, z]
      q0 (x): sims [9, 1, 2, 3] -> x at rank 1              hit@1,5,10
      q1 (y): sims [9, 8, 1, 2] -> first y at rank 2        hit@5,10 only
      q2 (z): sims [9, 8, 7, 1] -> z at rank 4              hit@5,10 only

      R@1 = 1/3, R@5 = R@10 = 3/3  ->  Rsum = 1/3 + 1 + 1 = 7/3
    """
    sim = np.array([[9, 1, 2, 3],
                    [9, 8, 1, 2],
                    [9, 8, 7, 1]], dtype=np.float32)
    m = rank_metrics(sim, np.array(["x", "y", "z"]),
                     np.array(["x", "y", "y", "z"]))
    assert m["R@1"] == pytest.approx(1 / 3)
    assert m["R@5"] == pytest.approx(1.0)
    assert m["Rsum"] == pytest.approx(m["R@1"] + m["R@5"] + m["R@10"])
    assert m["Rsum"] == pytest.approx(7 / 3)
    # papers print Rsum x100 (MGCC's Occluded-CUHK row is 233.40)
    assert 100 * m["Rsum"] == pytest.approx(233.333, abs=1e-2)


def test_rsum_absent_when_the_ks_cannot_form_it():
    sim = np.array([[1.0, 0.0]], dtype=np.float32)
    m = rank_metrics(sim, np.array(["a"]), np.array(["a", "b"]), ks=(1, 2))
    assert "Rsum" not in m


# --------------------------------------------------------------------------- #
# Per-image seeding -- portable across machines, and well mixed                #
# --------------------------------------------------------------------------- #
def test_seed_ignores_the_directory_prefix():
    """The same dataset-relative key must give the same occluder on a Windows
    workstation and on a Linux server, or "every baseline saw the same occluded
    gallery" is only true per machine."""
    from sar_person.occlude_bench import _seed_for
    assert _seed_for("test/0000/0000_013.jpg") == _seed_for("test/0000/0000_013.jpg")
    assert _seed_for("test/0000/0000_013.jpg") != _seed_for("test/0000/0000_014.jpg")


def test_seed_mixes_filenames_that_share_a_tail():
    """Regression: seeding from the last 8 bytes of the path collapsed ICFG's
    19 848 images onto 710 seeds, because the tail is just the frame index
    (`..._0015_0.jpg` -> `15_0.jpg`). That skewed the occluder class draw 5.8x
    against MGCC's uniform choice -- enough to miss the 57.24 gate check for a
    reason unrelated to the pipeline."""
    from sar_person.occlude_bench import _seed_for
    names = [f"test/{i // 20:04d}/{i // 20:04d}_{i % 20:03d}_0303morning_0015_0.jpg"
             for i in range(2000)]
    assert len({_seed_for(n) for n in names}) == 2000          # no collisions
    old = {int.from_bytes(n.encode()[-8:], "little") & 0x7FFFFFFF for n in names}
    assert len(old) < 50, "the old scheme should collapse -- if not, re-check"


def test_class_draw_is_near_uniform_like_mgcc(oil):
    """MGCC draws `random.choice(occlusion_list)` -- uniform over 15 classes.
    Ours must be too, or Occluded-* has a different occlusion mix than theirs."""
    import random as _r
    from collections import Counter
    from sar_person.occlude_bench import _seed_for
    names = [f"test/{i // 20:04d}/{i // 20:04d}_{i % 20:03d}_0015_0.jpg"
             for i in range(6000)]
    c = Counter(_r.Random(_seed_for(n)).choice(oil.classes) for n in names)
    assert len(c) == 15
    assert max(c.values()) / min(c.values()) < 1.5             # measured ~1.11


# --------------------------------------------------------------------------- #
# icfg vs icfg_mgcc -- two galleries that must never be averaged               #
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_icfg(tmp_path):
    """20 test ids x 5 images, laid out id-by-id like the real annotation."""
    import json
    d = tmp_path / "ICFG-PEDES" / "ICFG-PEDES"
    (d / "imgs").mkdir(parents=True)
    recs = [{"split": "train", "id": 999, "file_path": "tr.jpg",
             "captions": ["a train row"]}]
    for pid in range(20):
        for v in range(5):
            fp = f"{pid:04d}_{v}.jpg"
            Image.new("RGB", (8, 16)).save(d / "imgs" / fp)
            recs.append({"split": "test", "id": pid, "file_path": fp,
                         "captions": [f"person {pid} view {v}"]})
    (d / "ICFG-PEDES.json").write_text(json.dumps(recs), encoding="utf-8")
    return tmp_path


def test_full_icfg_keeps_every_test_record(fake_icfg):
    from sar_person.eval_sets import build
    es = build("icfg", fake_icfg)
    assert es.name == "icfg"
    assert len(es.gallery) == 100                    # train row excluded
    assert len(set(es.gallery_pids)) == 20


def test_mgcc_truncation_drops_identities_not_just_images(fake_icfg):
    """The point the paper never makes: MGCC truncates in *file order*, and the
    file is grouped by identity, so a 40% cut of the records is a 40% cut of the
    identities too. A smaller gallery AND fewer distractor ids both raise R@1 --
    that is why 59.28 cannot be read against a 19 848-image number."""
    from sar_person.eval_sets import _load_icfg
    es = _load_icfg(fake_icfg, truncate=40, name="icfg_mgcc")
    assert len(es.gallery) == 40
    assert len(set(es.gallery_pids)) == 8            # ids 0..7 only, not 20
    assert es.meta["protocol"] == "mgcc"
    assert es.meta["n_test_records_full"] == 100


def test_the_two_benchmarks_are_registered_separately(fake_icfg):
    from sar_person.eval_sets import BUILDERS, NONSTANDARD, build
    assert {"icfg", "icfg_mgcc"} <= set(BUILDERS)
    assert "icfg_mgcc" in NONSTANDARD and "icfg" not in NONSTANDARD
    # distinct names => distinct cache files => a result file always records
    # which gallery produced it
    assert build("icfg", fake_icfg).name != build("icfg_mgcc", fake_icfg).name


def test_mgcc_set_has_no_expected_block(fake_icfg):
    """`_sanity` warns when stats miss `expected`. The truncated set has no
    published protocol to be checked against, so claiming one would emit a
    warning that trains the reader to ignore warnings."""
    from sar_person.eval_sets import _load_icfg
    assert "expected" not in _load_icfg(fake_icfg, truncate=40,
                                        name="icfg_mgcc").meta
    assert "expected" in _load_icfg(fake_icfg).meta
