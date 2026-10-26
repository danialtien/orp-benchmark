"""Tests for the IRRA checkpoint adapter.

No torch, no CLIP weights, no IRRA repo: everything here covers the parts that
run BEFORE the model is built -- config resolution, checkpoint unwrapping,
identity-count recovery -- which is exactly where loading someone else's
checkpoint goes wrong.
"""
from __future__ import annotations

import sys

import pytest

from sar_person import irra_encoder as I


# --------------------------------------------------------------------------- #
# Cfg                                                                         #
# --------------------------------------------------------------------------- #

def test_cfg_supports_the_attribute_access_irra_expects():
    c = I.Cfg({"img_size": (384, 128), "stride_size": 12})
    assert c.img_size == (384, 128) and c.stride_size == 12
    c.temperature = 0.02
    assert c["temperature"] == 0.02


def test_cfg_raises_attribute_error_not_key_error():
    with pytest.raises(AttributeError):
        I.Cfg({}).nope


# --------------------------------------------------------------------------- #
# config loading                                                              #
# --------------------------------------------------------------------------- #

def _cfg_file(tmp_path, **kw):
    import yaml
    p = tmp_path / "configs.yaml"
    p.write_text(yaml.safe_dump(kw))
    return p


def test_config_is_read_verbatim(tmp_path):
    p = _cfg_file(tmp_path, pretrain_choice="ViT-B/16", stride_size=12,
                  img_size=[384, 128], loss_names="sdm+mlm+id")
    cfg = I.load_config(p)
    assert cfg.stride_size == 12                # NOT the 16 default
    assert cfg.img_size == (384, 128)           # list -> tuple for build_model
    assert cfg.training is False


def test_missing_keys_fall_back_to_irra_defaults(tmp_path, caplog):
    cfg = I.load_config(_cfg_file(tmp_path, pretrain_choice="ViT-B/16"))
    assert cfg.stride_size == I.DEFAULTS["stride_size"]
    assert cfg.cmt_depth == I.DEFAULTS["cmt_depth"]
    assert "using IRRA defaults" in caplog.text


def test_explicit_overrides_beat_the_file(tmp_path):
    p = _cfg_file(tmp_path, stride_size=12, img_size=[384, 128])
    cfg = I.load_config(p, {"stride_size": 16, "img_size": None})
    assert cfg.stride_size == 16
    assert cfg.img_size == (384, 128)           # None override is ignored


def test_a_missing_config_says_so(tmp_path):
    with pytest.raises(SystemExit, match="config not found"):
        I.load_config(tmp_path / "nope.yaml")


# --------------------------------------------------------------------------- #
# checkpoint discovery + unwrapping                                           #
# --------------------------------------------------------------------------- #

def test_best_pth_is_found_next_to_the_config(tmp_path):
    p = _cfg_file(tmp_path)
    (tmp_path / "best.pth").write_bytes(b"x")
    assert I.find_checkpoint(p) == tmp_path / "best.pth"


def test_explicit_checkpoint_wins(tmp_path):
    p = _cfg_file(tmp_path)
    (tmp_path / "best.pth").write_bytes(b"x")
    other = tmp_path / "epoch40.pth"
    other.write_bytes(b"y")
    assert I.find_checkpoint(p, str(other)) == other


def test_no_checkpoint_anywhere_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="best.pth"):
        I.find_checkpoint(_cfg_file(tmp_path))
    with pytest.raises(SystemExit, match="not found"):
        I.find_checkpoint(_cfg_file(tmp_path), str(tmp_path / "ghost.pth"))


@pytest.mark.parametrize("wrapper", ["model", "state_dict", "model_state_dict"])
def test_every_release_wrapper_is_unwrapped(wrapper):
    inner = {"base_model.visual.proj": 1}
    assert I.state_dict_of({wrapper: inner, "epoch": 60}) == inner


def test_a_bare_state_dict_passes_through():
    sd = {"base_model.visual.proj": 1}
    assert I.state_dict_of(sd) == sd


def test_dataparallel_prefix_is_stripped():
    got = I.state_dict_of({"model": {"module.base_model.x": 1, "module.y": 2}})
    assert set(got) == {"base_model.x", "y"}


def test_a_non_dict_checkpoint_is_rejected():
    with pytest.raises(SystemExit, match="unrecognised"):
        I.state_dict_of([1, 2, 3])


# --------------------------------------------------------------------------- #
# num_classes                                                                 #
# --------------------------------------------------------------------------- #

class _T:
    def __init__(self, *shape):
        self.shape = shape


@pytest.mark.parametrize("n", [11003, 3102, 3701])
def test_identity_count_comes_from_the_checkpoint_not_a_guess(n):
    """CUHK / ICFG / RSTP each have their own; guessing breaks the load."""
    assert I.infer_num_classes({"classifier.weight": _T(n, 512)}) == n


def test_a_checkpoint_without_a_classifier_falls_back_and_warns(caplog):
    assert I.infer_num_classes({"base_model.x": _T(512, 512)}, 3102) == 3102
    assert "no classifier.weight" in caplog.text


# --------------------------------------------------------------------------- #
# repo path                                                                   #
# --------------------------------------------------------------------------- #

def test_missing_irra_repo_names_the_flag(tmp_path):
    with pytest.raises(SystemExit, match="--irra-root"):
        I.add_irra_to_path(tmp_path / "not-irra")


def test_repo_root_is_put_on_sys_path(tmp_path):
    root = tmp_path / "IRRA"
    (root / "model").mkdir(parents=True)
    (root / "model" / "build.py").write_text("")
    before = list(sys.path)
    try:
        assert I.add_irra_to_path(root) == root
        assert str(root) in sys.path
        I.add_irra_to_path(root)                       # idempotent
        assert sys.path.count(str(root)) == 1
    finally:
        sys.path[:] = before


# --------------------------------------------------------------------------- #
# wiring                                                                      #
# --------------------------------------------------------------------------- #

def test_eval_protocol_demands_a_config_before_touching_torch():
    from sar_person.eval_protocol import get_eval_encoder
    with pytest.raises(SystemExit, match="--irra-config"):
        get_eval_encoder("irra")


def test_the_cli_rejects_a_missing_config_before_building_the_eval_set(
        monkeypatch, capsys):
    """Building the ICFG eval set takes minutes; the flag check must come first."""
    import sar_person.eval_protocol as EP

    def explode(*a, **kw):                       # pragma: no cover
        raise AssertionError("the eval set must not be built")

    monkeypatch.setattr(EP, "build_evalset", explode)
    monkeypatch.setattr(sys, "argv",
                        ["eval_protocol", "--benchmark", "rstp",
                         "--encoder", "irra", "--out", "/tmp/x"])
    with pytest.raises(SystemExit, match="--irra-config"):
        EP.main()


def test_l2_normalises_rows():
    import numpy as np
    got = I._l2(np.array([[3.0, 4.0], [0.0, 2.0]]))
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-6)
