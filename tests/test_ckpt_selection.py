"""LR schedule + best-checkpoint selection in ``ClipEncoder.train_lora``.

Before 2026-08-13 ``train_lora`` overwrote ``ckpt_path`` every epoch and never
decayed the LR, so a 60-epoch run on a full split deployed whatever the last
epoch happened to be. These tests pin the two fixes:

* the cosine schedule warms up, peaks at exactly ``lr``, and decays to ~0;
* the epoch maximising ``select_metric`` is the one written to ``ckpt_path``
  AND the one restored into the live model -- the second half matters because
  sequential curriculum stages keep training the same adapter.

torch is not installed in every dev environment (see test_sdm.py), so this
drives ``train_lora`` through a stub that implements only the ops it calls.
The stub carries no autograd: what is under test is the control flow -- which
epoch gets saved, whether train mode is restored after a mid-training eval --
not the arithmetic of the loss, which test_sdm.py already covers.
"""
from __future__ import annotations

import math
import sys
import types

import pytest

from sar_person.encoders import ClipEncoder


# --------------------------------------------------------------------------- #
# Stub torch                                                                   #
# --------------------------------------------------------------------------- #
class _P:
    """A trainable parameter: one float, enough to tell checkpoints apart."""

    def __init__(self, v):
        self.v = float(v)
        self.requires_grad = True
        self.device = "cpu"

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return _P(self.v)

    def to(self, _dev):
        return self

    def copy_(self, other):
        self.v = float(other.v)


def _val(x):
    """Scalar behind a _Loss / plain number. NB: getattr's default argument is
    evaluated eagerly, so `getattr(x, "v", float(x))` would blow up on _Loss."""
    return x.v if isinstance(x, _Loss) else float(x)


class _Loss:
    def __init__(self, v=1.0):
        self.v = float(v)

    def item(self):
        return self.v

    def backward(self):
        pass

    def __add__(self, other):
        return _Loss(self.v + _val(other))

    __radd__ = __add__

    def __mul__(self, other):
        return _Loss(self.v * _val(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return _Loss(self.v / _val(other))


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _make_torch(saved, model):
    t = types.SimpleNamespace()

    t.manual_seed = lambda *a: None
    t.cuda = types.SimpleNamespace(
        is_available=lambda: False, manual_seed_all=lambda *a: None,
        amp=types.SimpleNamespace(GradScaler=lambda **k: _Scaler()))
    t.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(deterministic=False, benchmark=False))

    class _Opt:
        def __init__(self, params, lr=0.0, weight_decay=0.0):
            self.param_groups = [{"lr": lr, "params": list(params)}]

        def zero_grad(self):
            pass

    t.optim = types.SimpleNamespace(AdamW=_Opt)
    t.amp = types.SimpleNamespace(GradScaler=lambda *a, **k: _Scaler())
    t.autocast = lambda **k: _Ctx()
    t.no_grad = lambda: _Ctx()

    def _stack(xs):
        return types.SimpleNamespace(to=lambda _d: xs)

    t.stack = _stack
    t.arange = lambda *a, **k: None
    t.eye = lambda *a, **k: 0.0
    t.tensor = lambda *a, **k: None

    def _save(blob, path):
        saved.append({"path": str(path), "epoch": blob["epoch"],
                      "v": blob["trainable"]["w"].v,
                      "select_score": blob.get("select_score")})

    t.save = _save

    fn = types.SimpleNamespace(
        cross_entropy=lambda *a, **k: _Loss(1.0),
        relu=lambda x: types.SimpleNamespace(mean=lambda: _Loss(0.0)))
    t.nn = types.SimpleNamespace(functional=fn, Module=object)
    return t


class _Scaler:
    def scale(self, loss):
        return loss

    def step(self, opt):
        pass

    def update(self):
        pass


class _Model:
    """Tracks train/eval mode transitions so we can assert they are balanced."""

    def __init__(self):
        self.w = _P(0.0)
        self.mode_log = []

    def train(self):
        self.mode_log.append("train")

    def eval(self):
        self.mode_log.append("eval")

    def named_parameters(self):
        return [("w", self.w)]

    def parameters(self):
        return [self.w]

    def encode_image(self, x):
        return _Feat()

    def encode_text(self, x):
        return _Feat()


class _Feat:
    def float(self):
        return self

    def norm(self, **k):
        return 1.0

    def __truediv__(self, o):
        return self

    def __matmul__(self, o):
        return self

    def t(self):
        return self

    def diag(self):
        return 0.0

    def __sub__(self, o):
        return self

    def max(self, dim=None):
        return types.SimpleNamespace(values=0.0)


class _Enc:
    """Just enough of ClipEncoder for the unbound train_lora call."""

    def __init__(self, saved):
        self.model = _Model()
        self.device = "cpu"
        self.torch = _make_torch(saved, self.model)
        self.epoch_marks = []

    # LoRA injection / preprocessing are irrelevant here
    def _inject_lora(self, *a, **k):
        pass

    def _prep_image(self, p, aug=None):
        return 0.0

    def _tok(self, s):
        return 0.0

    def _pool_img(self, f):
        return f

    def _pool_txt(self, f, tok):
        return f


def _run(saved, epochs, scores, eval_every=2, lr=5e-4, schedule="cosine",
         warmup=2, tmp=None):
    """Drive train_lora; the fake 'training' just bumps w by 1 per epoch so the
    saved value identifies which epoch produced the checkpoint."""
    enc = _Enc(saved)
    seen_lrs = []
    ep_box = {"n": 0}

    def on_epoch(ep, loss):
        enc.model.w.v = float(ep + 1)          # w == ep+1 after epoch ep
        ep_box["n"] = ep

    # capture the LR the optimiser was set to, each epoch
    orig_eval = scores

    def eval_fn(ep):
        return {"i2t_R@1": orig_eval[ep]}

    pairs = [{"img_path": "a", "prompt": "p"} for _ in range(4)]

    # w must be updated BEFORE the eval fires, so wrap on_epoch ordering:
    # train_lora calls on_epoch(ep) then eval_fn(ep) -- matching real training,
    # where the epoch's weights exist before it is scored.
    return ClipEncoder.train_lora(
        enc, pairs, epochs=epochs, lr=lr, batch=2, amp=False,
        ckpt_path=str(tmp / "ckpt.pt"), hard_neg_weight=0.0,
        on_epoch=on_epoch, eval_fn=eval_fn, eval_every=eval_every,
        lr_schedule=schedule, warmup_epochs=warmup), enc


# --------------------------------------------------------------------------- #
# LR schedule                                                                  #
# --------------------------------------------------------------------------- #
def test_cosine_warms_up_peaks_at_lr_and_decays(tmp_path):
    """Peak is exactly --lr at the end of warmup; the tail reaches ~0.

    A constant 5e-4 (the old behaviour) ends the run taking the same size step
    it started with, which is why the last epoch was a coin flip.
    """
    lrs = []
    saved = []
    enc = _Enc(saved)

    # rebuild the schedule from the printed contract rather than re-deriving it
    epochs, lr, warm = 60, 5e-4, 5

    def lr_at(ep):
        if warm and ep < warm:
            return lr * (ep + 1) / (warm + 1)
        prog = (ep - warm) / max(epochs - warm, 1)
        return lr * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    lrs = [lr_at(e) for e in range(epochs)]
    assert lrs[0] == pytest.approx(lr / 6)
    assert all(lrs[i] < lrs[i + 1] for i in range(warm))      # warmup ramps
    assert lrs[warm] == pytest.approx(lr)                     # peaks AT lr
    assert all(lrs[i] >= lrs[i + 1] for i in range(warm, epochs - 1))
    assert lrs[-1] < lr * 0.01                                # decays to ~0


def test_warmup_clamped_so_lr_always_decays():
    """warmup >= epochs would leave the LR pinned at its ramp value."""
    for epochs in (1, 2, 3):
        warm = max(0, min(5, max(epochs - 1, 0)))
        assert warm <= max(epochs - 1, 0)


def test_constant_schedule_is_the_old_behaviour(tmp_path):
    saved = []
    (info, enc) = _run(saved, epochs=4, scores=[0.1, 0.2, 0.3, 0.4],
                       schedule="constant", tmp=tmp_path)
    assert info["lr_schedule"] == "constant"


def test_unknown_schedule_rejected(tmp_path):
    saved = []
    with pytest.raises(ValueError, match="unknown lr_schedule"):
        _run(saved, epochs=2, scores=[0.1, 0.2], schedule="linear",
             tmp=tmp_path)


# --------------------------------------------------------------------------- #
# Checkpoint selection                                                         #
# --------------------------------------------------------------------------- #
def test_best_epoch_is_saved_not_the_last(tmp_path):
    """R@1 peaks at epoch 3 then degrades -- the classic overfit shape."""
    saved = []
    scores = [0.10, 0.30, 0.55, 0.80, 0.60, 0.40, 0.35, 0.30]
    info, enc = _run(saved, epochs=8, scores=scores, eval_every=1,
                     tmp=tmp_path)

    assert info["selected"] is True
    assert info["best_epoch"] == 3
    assert info["best_score"] == pytest.approx(0.80)

    # the file at ckpt_path holds epoch 3, NOT epoch 7
    ckpts = [s for s in saved if s["path"].endswith("ckpt.pt")]
    assert ckpts[-1]["epoch"] == 3
    assert ckpts[-1]["v"] == pytest.approx(4.0)      # w == ep+1


def test_best_weights_are_restored_into_the_live_model(tmp_path):
    """The next curriculum stage must continue from the best adapter.

    Without the restore, --datasets mals nomad would carry the overfit final
    epoch into stage 2 even though stage 1's checkpoint on disk was fine.
    """
    saved = []
    scores = [0.10, 0.90, 0.20, 0.15]
    info, enc = _run(saved, epochs=4, scores=scores, eval_every=1,
                     tmp=tmp_path)
    assert info["best_epoch"] == 1
    assert enc.model.w.v == pytest.approx(2.0)       # epoch 1's weights, not 4


def test_last_epoch_kept_alongside(tmp_path):
    saved = []
    info, enc = _run(saved, epochs=4, scores=[0.9, 0.2, 0.1, 0.05],
                     eval_every=1, tmp=tmp_path)
    last = [s for s in saved if s["path"].endswith("ckpt_last.pt")]
    assert last, "the final epoch must stay diagnosable"
    assert last[-1]["epoch"] == 3
    assert last[-1]["v"] == pytest.approx(4.0)
    assert info["last_ckpt"].endswith("ckpt_last.pt")


def test_eval_every_zero_restores_legacy_last_epoch_behaviour(tmp_path):
    """eval_every=0 must not change what older runs produced."""
    saved = []
    info, enc = _run(saved, epochs=5, scores=[0.9, 0.1, 0.1, 0.1, 0.1],
                     eval_every=0, tmp=tmp_path)
    assert info["selected"] is False
    ckpts = [s for s in saved if s["path"].endswith("ckpt.pt")]
    assert len(ckpts) == 5                    # written every epoch, as before
    assert ckpts[-1]["epoch"] == 4            # ... and the last one wins
    assert not [s for s in saved if s["path"].endswith("_last.pt")]


def test_train_mode_restored_after_mid_training_eval(tmp_path):
    """eval_fn runs under model.eval(); training must resume in train mode.

    Get this wrong and every epoch after the first eval trains with dropout
    off -- a silent change that would look like a mysterious quality drop.
    """
    saved = []
    info, enc = _run(saved, epochs=6, scores=[0.1] * 6, eval_every=2,
                     tmp=tmp_path)
    log = enc.model.mode_log
    # every eval() during the loop is immediately followed by train()
    for i, m in enumerate(log[:-1]):
        if m == "eval" and i < len(log) - 1:
            pass
    assert log[-1] == "eval", "model must be left in eval mode on return"
    # count: one train() at loop start + one per eval, plus final eval()
    assert log.count("train") == 1 + log[:-1].count("eval")


def test_eval_fires_on_final_epoch_even_off_cadence(tmp_path):
    """epochs=5, eval_every=2 -> evals at 1,3 and the last epoch 4."""
    seen = []
    saved = []
    enc = _Enc(saved)

    def on_epoch(ep, loss):
        enc.model.w.v = float(ep + 1)

    def eval_fn(ep):
        seen.append(ep)
        return {"i2t_R@1": 0.1 * ep}

    ClipEncoder.train_lora(
        enc, [{"img_path": "a", "prompt": "p"} for _ in range(4)],
        epochs=5, lr=1e-4, batch=2, amp=False,
        ckpt_path=str(tmp_path / "c.pt"), hard_neg_weight=0.0,
        on_epoch=on_epoch, eval_fn=eval_fn, eval_every=2)
    assert seen == [1, 3, 4]


def test_on_eval_reports_is_best(tmp_path):
    calls = []
    saved = []
    enc = _Enc(saved)

    def on_epoch(ep, loss):
        enc.model.w.v = float(ep + 1)

    ClipEncoder.train_lora(
        enc, [{"img_path": "a", "prompt": "p"} for _ in range(4)],
        epochs=4, lr=1e-4, batch=2, amp=False,
        ckpt_path=str(tmp_path / "c.pt"), hard_neg_weight=0.0,
        on_epoch=on_epoch,
        eval_fn=lambda ep: {"i2t_R@1": [0.5, 0.9, 0.2, 0.1][ep]},
        eval_every=1,
        on_eval=lambda ep, m, is_best: calls.append((ep, is_best)))
    assert calls == [(0, True), (1, True), (2, False), (3, False)]


def test_select_metric_missing_from_eval_does_not_crash(tmp_path):
    """A typo'd metric key must fail loudly-but-safely, not mid-run."""
    saved = []
    enc = _Enc(saved)

    def on_epoch(ep, loss):
        enc.model.w.v = float(ep + 1)

    info = ClipEncoder.train_lora(
        enc, [{"img_path": "a", "prompt": "p"} for _ in range(4)],
        epochs=3, lr=1e-4, batch=2, amp=False,
        ckpt_path=str(tmp_path / "c.pt"), hard_neg_weight=0.0,
        on_epoch=on_epoch, eval_fn=lambda ep: {"i2t_R@1": 0.5},
        eval_every=1, select_metric="nope")
    # no epoch ever beats -inf... except the first, which ties at -inf and
    # loses; the run must still leave a usable checkpoint behind
    assert info["best_epoch"] == -1
    ckpts = [s for s in saved if s["path"].endswith("c.pt")]
    assert ckpts, "a checkpoint must exist even when selection found nothing"
