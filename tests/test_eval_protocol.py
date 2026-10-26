"""Hand-computable checks for the paper-grade retrieval metrics.

Every case here has an answer worked out by hand in the docstring, because the
whole point of eval_protocol is that its numbers are trustworthy enough to put
next to published tables.
"""
import numpy as np
import pytest

from sar_person.eval_protocol import rank_metrics, paired_bootstrap
from sar_person.eval_sets import EvalSet, parse_cuhk_pid


def test_single_query_two_positives():
    """Gallery ranked [pid1, pid2, pid1, pid3] for a pid-1 query.

    positives at 1-based ranks 1 and 3
      AP   = mean(1/1, 2/3)  = 0.83333
      INP  = 2/3             = 0.66667
      R@1  = 1
    """
    sim = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    g = np.array(["1", "2", "1", "3"])
    q = np.array(["1"])
    m = rank_metrics(sim, q, g, ks=(1, 2, 3))
    assert m["R@1"] == pytest.approx(1.0)
    assert m["mAP"] == pytest.approx(0.8333333, abs=1e-6)
    assert m["mINP"] == pytest.approx(0.6666667, abs=1e-6)
    assert m["n_gallery"] == 4 and m["n_query"] == 1


def test_perfect_and_worst_ranking():
    """All positives first -> AP = INP = 1. All positives last -> INP is small."""
    g = np.array(["a", "a", "b", "b"])
    q = np.array(["a"])
    best = rank_metrics(np.array([[1.0, 0.9, 0.1, 0.0]], np.float32), q, g)
    assert best["mAP"] == pytest.approx(1.0) and best["mINP"] == pytest.approx(1.0)
    # positives pushed to ranks 3 and 4: AP = mean(1/3, 2/4) = 0.41667, INP = 2/4
    worst = rank_metrics(np.array([[0.0, 0.1, 1.0, 0.9]], np.float32), q, g)
    assert worst["mAP"] == pytest.approx(0.4166667, abs=1e-6)
    assert worst["mINP"] == pytest.approx(0.5)


def test_recall_is_not_precision():
    """R@5 must be 1 as soon as ONE positive is in the top 5, not the fraction."""
    g = np.array(["a"] + ["b"] * 9)
    q = np.array(["a"])
    sim = np.zeros((1, 10), np.float32)
    sim[0, 0] = 0.5                      # the single positive ranks 1st
    m = rank_metrics(sim, q, g, ks=(1, 5, 10))
    assert m["R@1"] == 1.0 and m["R@5"] == 1.0 and m["R@10"] == 1.0


def test_unanswerable_query_is_excluded_not_counted_as_zero():
    """A query whose pid is absent from the gallery would cap every metric."""
    g = np.array(["a", "a"])
    q = np.array(["a", "zzz"])
    sim = np.array([[1.0, 0.5], [0.2, 0.1]], np.float32)
    m = rank_metrics(sim, q, g)
    assert m["n_query"] == 1 and m["n_query_skipped"] == 1
    assert m["mAP"] == pytest.approx(1.0)


def test_fp16_similarity_is_rejected():
    with pytest.raises(TypeError):
        rank_metrics(np.zeros((1, 2), np.float16), np.array(["a"]),
                     np.array(["a", "b"]))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        rank_metrics(np.zeros((1, 3), np.float32), np.array(["a"]),
                     np.array(["a", "b"]))


def test_ties_are_resolved_deterministically():
    g = np.array(["a", "b", "a", "b"])
    q = np.array(["a"])
    sim = np.full((1, 4), 0.5, np.float32)          # every score identical
    first = rank_metrics(sim, q, g)
    for _ in range(5):
        assert rank_metrics(sim, q, g) == first


def test_metrics_match_a_known_two_query_case():
    """Two queries, worked out by hand.

    q0 (pid a): ranking a,b,a  -> AP = mean(1/1, 2/3) = 0.83333, INP = 2/3
    q1 (pid b): ranking a,a,b  -> AP = 1/3,             INP = 1/3
      mAP  = (0.83333 + 0.33333)/2 = 0.58333
      mINP = (0.66667 + 0.33333)/2 = 0.5
      R@1  = (1 + 0)/2 = 0.5
    """
    g = np.array(["a", "b", "a"])
    sim = np.array([[0.9, 0.8, 0.7],          # q0 -> a, b, a
                    [0.9, 0.1, 0.8]], np.float32)  # q1 -> a, a, b
    q = np.array(["a", "b"])
    m = rank_metrics(sim, q, g, ks=(1,))
    assert m["R@1"] == pytest.approx(0.5)
    assert m["mAP"] == pytest.approx(0.5833333, abs=1e-6)
    assert m["mINP"] == pytest.approx(0.5, abs=1e-6)


def test_paired_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, 500)
    b = rng.normal(0.5, 0.05, 500)
    r = paired_bootstrap(a, b, n_boot=2000, seed=0)
    assert r["delta"] > 0.2 and r["p_value"] < 0.05
    assert r["ci95_low"] > 0


def test_paired_bootstrap_finds_no_difference_in_noise():
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 0.2, 400)
    b = a + rng.normal(0.0, 0.001, 400)
    r = paired_bootstrap(a, b, n_boot=2000, seed=0)
    assert r["p_value"] > 0.05
    assert r["ci95_low"] < 0 < r["ci95_high"]


# --------------------------------------------------------------------------- #
# EvalSet plumbing                                                             #
# --------------------------------------------------------------------------- #
def _toy_evalset():
    gallery = [{"image": f"/tmp/img{i}.jpg", "pid": str(i % 2),
                "image_id": str(i), "visibility": 20 + 30 * i}
               for i in range(4)]
    queries = [{"text": f"person {p}", "pid": str(p), "image_id": f"q{p}"}
               for p in range(2)]
    return EvalSet("toy", gallery, queries, {})


def test_evalset_stats_and_roundtrip(tmp_path):
    es = _toy_evalset()
    st = es.stats()
    assert st["gallery_images"] == 4 and st["gallery_ids"] == 2
    assert st["queries"] == 2 and st["queries_without_gallery_match"] == 0
    p = es.save(tmp_path / "toy.json")
    back = EvalSet.load(p)
    assert back.gallery == es.gallery and back.queries == es.queries


def test_evalset_remap_keeps_queries_and_pids():
    es = _toy_evalset()
    mapping = {g["image"]: g["image"].replace("/tmp", "/occ") for g in es.gallery}
    d = es.with_gallery_images(mapping, "toy_torso_l50", {"variant": "torso"})
    assert [g["pid"] for g in d.gallery] == [g["pid"] for g in es.gallery]
    assert d.queries == es.queries
    assert all(g["image"].startswith("/occ") for g in d.gallery)
    assert d.gallery[0]["source_image"] == es.gallery[0]["image"]


def test_evalset_remap_drops_unmapped_images():
    es = _toy_evalset()
    partial = {es.gallery[0]["image"]: "/occ/a.jpg"}
    d = es.with_gallery_images(partial, "partial")
    assert len(d.gallery) == 1


def test_orphan_query_is_reported():
    es = _toy_evalset()
    es.queries.append({"text": "ghost", "pid": "999", "image_id": "q9"})
    assert es.stats()["queries_without_gallery_match"] == 1


# --------------------------------------------------------------------------- #
# CUHK-PEDES identity recovery from filenames                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,expected", [
    ("0363001.png", "cuhk01:0363"),              # CUHK01: 4-digit pid + 3-digit idx
    ("0001004.png", "cuhk01:0001"),
    ("1407_c3s3_049303_00.jpg", "market:1407"),  # Market-1501
    ("-1_c2s1_000301_00.jpg", "market:-1"),      # Market junk id, still parsed
    ("p10082_s15238.jpg", "sysu:10082"),         # CUHK-SYSU
    ("339_90.bmp", "viper:339"),                 # VIPeR
    ("1179_1.jpg", "cuhk03:1179"),               # CUHK03
    ("1488_c3s3_071419_00.jpg.jpg", "market:1488"),   # doubled suffix in HF export
    ("totally_unexpected_name.tiff", None),
])
def test_parse_cuhk_pid(name, expected):
    assert parse_cuhk_pid(name) == expected


def test_cuhk_pid_namespaces_do_not_collide():
    """Market 339 and VIPeR 339 are different people; a bare int would fuse them."""
    assert parse_cuhk_pid("339_90.bmp") != parse_cuhk_pid("339_c1s1_000001_00.jpg")


def test_parse_cuhk_pid_accepts_a_full_path():
    assert parse_cuhk_pid("train/CUHK01/0363001.png") == "cuhk01:0363"
    assert parse_cuhk_pid(r"some\windows\path\p10082_s15238.jpg") == "sysu:10082"
