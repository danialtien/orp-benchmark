"""Tests for the actor-level NOMAD evaluation. No GPU, no CLIP weights."""
from __future__ import annotations

import json

import numpy as np
import pytest

from sar_person import eval_nomad_actors as E


# --------------------------------------------------------------------------- #
# data loading                                                                #
# --------------------------------------------------------------------------- #

def _crop_set(tmp_path, actors=2, per_actor=6, altitudes=(10, 30)):
    root = tmp_path / "crop"
    (root / "images").mkdir(parents=True)
    recs, profiles = [], {}
    for a in range(1, actors + 1):
        actor = f"Actor{a:03d}"
        (root / "images" / actor).mkdir()
        for alt in altitudes:
            for f in range(per_actor):
                tag = f"{actor}_a{alt}_f{f:04d}_p0"
                recs.append({
                    "id": tag, "file_path": f"{actor}/{tag}.jpg",
                    "person_id": actor, "split": "val" if a % 5 == 0 else "train",
                    "altitude_m": alt, "visibility": 50 + f * 5, "frame": f"{f:04d}",
                    "captions": [f"caption {a}"],
                })
        profiles[actor] = {"prompt_default": f"a person number {a}",
                           "prompt_templated": f"the person is number {a}"}
    (root / "annotations.json").write_text(json.dumps(recs))
    (root / "nomad_profiles.json").write_text(json.dumps(profiles))
    return root


def test_crops_are_grouped_by_actor(tmp_path):
    root = _crop_set(tmp_path, actors=3, per_actor=4)
    got = E.load_actor_crops(root)
    assert sorted(got) == ["Actor001", "Actor002", "Actor003"]
    assert len(got["Actor001"]) == 8                    # 4 frames x 2 altitudes
    assert all(p.endswith(".jpg") for p in got["Actor001"])


def test_altitude_and_visibility_filters(tmp_path):
    root = _crop_set(tmp_path, actors=2, per_actor=6)
    assert len(E.load_actor_crops(root, altitudes=[10])["Actor001"]) == 6
    # visibility runs 50,55,60,... -> a floor of 65 keeps the last 3 of each 6
    got = E.load_actor_crops(root, altitudes=[10], min_visibility=65)
    assert len(got["Actor001"]) == 3


def test_split_filter_selects_the_holdout(tmp_path):
    root = _crop_set(tmp_path, actors=6, per_actor=2)
    got = E.load_actor_crops(root, split="val")
    assert sorted(got) == ["Actor005"]                  # a % 5 == 0


def test_max_per_actor_samples_across_the_flight_not_one_window(tmp_path):
    root = _crop_set(tmp_path, actors=1, per_actor=20, altitudes=(10,))
    got = E.load_actor_crops(root, altitudes=[10], max_per_actor=4)["Actor001"]
    assert len(got) == 4
    frames = [int(p.split("_f")[1][:4]) for p in got]
    assert frames == sorted(frames)
    assert max(frames) - min(frames) >= 12              # spread, not consecutive


def test_missing_annotations_fails_loudly(tmp_path):
    (tmp_path / "crop").mkdir()
    with pytest.raises(SystemExit, match="prepare_nomad"):
        E.load_actor_crops(tmp_path / "crop")


def test_prompts_can_come_from_the_vlm_profiles_file(tmp_path):
    root = _crop_set(tmp_path, actors=2)
    vlm = root / "vlm"
    vlm.mkdir()
    (vlm / "nomad_vlm_profiles.json").write_text(json.dumps(
        {"Actor001": {"prompt_default": "vlm caption one"},
         "Actor002": {"prompt_default": "vlm caption two"}}))
    got = E.load_actor_prompts(root, profiles=str(vlm / "nomad_vlm_profiles.json"))
    assert got["Actor001"] == "vlm caption one"
    assert E.load_actor_prompts(root)["Actor001"] == "a person number 1"


def test_prompt_style_selects_the_flavour(tmp_path):
    root = _crop_set(tmp_path, actors=2)
    got = E.load_actor_prompts(root, prompt_style="templated")
    assert got["Actor001"] == "the person is number 1"


# --------------------------------------------------------------------------- #
# prototypes                                                                  #
# --------------------------------------------------------------------------- #

class _Enc:
    """Encoder stub: each crop path maps to a fixed vector via its actor name."""

    def __init__(self, table, noise=0.0, seed=0):
        self.table = table
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.n_image_calls = 0

    def encode_images(self, paths, batch=32):
        self.n_image_calls += 1
        out = []
        for p in paths:
            actor = str(p).split("/")[-1].split("\\")[-1].split("_")[0]
            v = np.asarray(self.table[actor], dtype=np.float32)
            if self.noise:
                v = v + self.rng.normal(0, self.noise, v.shape)
            out.append(v / (np.linalg.norm(v) + 1e-8))
        return np.stack(out)

    def encode_texts(self, texts, batch=32):
        return np.stack([np.asarray(self.table[t], dtype=np.float32)
                         for t in texts])


def test_prototype_is_the_normalised_mean_of_the_crops():
    enc = _Enc({"Actor001": [1.0, 0.0], "Actor002": [0.0, 2.0]})
    crops = {"Actor001": ["d/Actor001_a10_f0_p0.jpg"] * 3,
             "Actor002": ["d/Actor002_a10_f0_p0.jpg"] * 2}
    P = E.actor_prototypes(enc, ["Actor001", "Actor002"], crops)
    assert P.shape == (2, 2)
    np.testing.assert_allclose(P[0], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(P[1], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(P, axis=1), 1.0, atol=1e-6)


def test_all_crops_are_encoded_in_one_call():
    enc = _Enc({f"Actor{i:03d}": [1.0, 0.0] for i in range(1, 6)})
    crops = {f"Actor{i:03d}": [f"d/Actor{i:03d}_a10_f0_p0.jpg"] * 4
             for i in range(1, 6)}
    E.actor_prototypes(enc, sorted(crops), crops)
    assert enc.n_image_calls == 1


def test_averaging_survives_one_bad_crop():
    """A single outlier crop must move the prototype, not decide it."""
    enc = _Enc({"Actor001": [1.0, 0.0]})
    good = ["d/Actor001_a10_f0_p0.jpg"] * 19
    P_clean = E.actor_prototypes(enc, ["Actor001"], {"Actor001": good})
    noisy = _Enc({"Actor001": [1.0, 0.0]}, noise=0.0)
    # one crop pointing 90 degrees away
    emb = np.concatenate([np.tile([1.0, 0.0], (19, 1)), [[0.0, 1.0]]])
    proto = emb.mean(0)
    proto /= np.linalg.norm(proto)
    assert float(P_clean[0] @ proto) > 0.99
    del noisy


# --------------------------------------------------------------------------- #
# scoring                                                                     #
# --------------------------------------------------------------------------- #

def test_perfect_matrix_scores_one():
    S = np.eye(4, dtype=np.float32)
    m = E.score(S, [f"caption {i}" for i in range(4)])
    assert m["t2i_R@1"] == 1.0 and m["i2t_R@1"] == 1.0
    assert m["t2i_MRR"] == 1.0 and m["t2i_median_rank"] == 1.0
    assert m["n_unique_captions"] == 4
    assert m["n_actors_sharing_a_caption"] == 0


def test_both_directions_are_reported_and_can_differ():
    # actor 0's prototype is closest to caption 1, but caption 1 is closest to
    # actor 1 -- i2t is wrong while t2i is right.
    S = np.array([[0.5, 0.6], [0.1, 0.9]], dtype=np.float32)
    m = E.score(S, ["c0", "c1"])
    assert m["i2t_R@1"] == 0.5
    assert m["t2i_R@1"] == 1.0


def test_duplicate_captions_are_counted_and_credited_separately():
    S = np.array([[0.9, 0.8, 0.1], [0.8, 0.9, 0.1], [0.1, 0.1, 0.9]],
                 dtype=np.float32)
    same = ["a man in a black top", "a man in a black top", "a woman in red"]
    m = E.score(S, same)
    assert m["n_unique_captions"] == 2
    assert m["n_actors_sharing_a_caption"] == 2
    assert m["t2i_R@1_caption"] >= m["t2i_R@1"]
    assert m["t2i_R@1_caption"] == 1.0        # every top-1 has the right TEXT


def test_median_rank_reports_where_the_truth_landed():
    n = 5
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        S[i] = np.linspace(1.0, 0.0, n)
        S[i, i] = 0.5                         # own caption always mid-table
    m = E.score(S, [f"c{i}" for i in range(n)])
    assert 1 < m["t2i_median_rank"] <= n


def test_score_keeps_the_run_py_metric_schema():
    S = np.eye(3, dtype=np.float32)
    m = E.score(S, ["a", "b", "c"])
    for k in ("top1_diagonal_acc", "mean_positive_sim", "mean_negative_sim",
              "pos_neg_gap", "i2t_MRR"):
        assert k in m


def test_caption_groups_are_by_exact_text():
    g = E.caption_groups(["x", "y", "x"])
    assert g[0] == g[2] and g[0] != g[1]


# --------------------------------------------------------------------------- #
# per-actor diagnostics                                                       #
# --------------------------------------------------------------------------- #

def test_per_actor_row_names_the_confuser():
    S = np.array([[0.5, 0.9], [0.4, 0.8]], dtype=np.float32)
    actors = ["Actor001", "Actor002"]
    crops = {a: ["x.jpg"] * 7 for a in actors}
    rows = E.per_actor_rows(S, actors, ["c0", "c1"], crops)
    # caption 0 ranks Actor001 (0.5) above Actor002 (0.4) -> rank 1
    assert rows[0]["t2i_rank"] == 1 and rows[0]["top1_actor"] == "Actor001"
    # caption 1 prefers Actor001 (0.9) over its true Actor002 (0.8) -> rank 2
    assert rows[1]["t2i_rank"] == 2 and rows[1]["top1_actor"] == "Actor001"
    assert rows[1]["margin"] < 0
    assert rows[0]["n_crops"] == 7


def test_per_actor_row_lists_caption_twins():
    S = np.eye(3, dtype=np.float32)
    actors = ["A", "B", "C"]
    rows = E.per_actor_rows(S, actors, ["same", "same", "other"],
                            {a: ["x.jpg"] for a in actors})
    assert rows[0]["caption_shared_with"] == ["B"]
    assert rows[2]["caption_shared_with"] == []
