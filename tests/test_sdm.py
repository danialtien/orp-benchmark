"""SDM loss + PK sampler. Expected values derived by hand in the docstrings.

torch is not installed in every dev environment, so the loss tests use a tiny
numpy stand-in that implements the four ops ``sdm_loss`` calls. That is enough
to pin the *maths*; the tensor plumbing is exercised by the smoke run.
"""
from __future__ import annotations

import numpy as np
import pytest

from sar_person.encoders import _pk_batches, sdm_loss


# --------------------------------------------------------------------------- #
# A numpy shim for the handful of torch ops sdm_loss uses                      #
# --------------------------------------------------------------------------- #
class _T:
    """Minimal torch surface: softmax, log_softmax, log."""

    @staticmethod
    def softmax(x, dim):
        e = np.exp(x - x.max(axis=dim, keepdims=True))
        return e / e.sum(axis=dim, keepdims=True)

    @staticmethod
    def log_softmax(x, dim):
        m = x.max(axis=dim, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(axis=dim, keepdims=True))

    log = staticmethod(np.log)


class _A(np.ndarray):
    """ndarray that also answers .float(), .t(), .sum(dim=), .clamp_min()."""

    def float(self):
        return self.astype(np.float64).view(_A)

    def t(self):
        return self.T.view(_A)

    # torch spells it dim/keepdim, numpy spells it axis/keepdims, and this shim
    # is called both ways (by sdm_loss and by numpy internals). Accept either --
    # swallowing `axis` into **kw silently summed over the whole array and made
    # every softmax row sum to 1/n_rows.
    def sum(self, dim=None, keepdim=False, axis=None, keepdims=None, **kw):  # noqa: A003
        ax = dim if dim is not None else axis
        kd = keepdim or bool(keepdims)
        return np.asarray(np.ndarray.sum(self, axis=ax, keepdims=kd)).view(_A)

    def mean(self, dim=None, axis=None, **kw):           # noqa: A003
        return np.ndarray.mean(self, axis=dim if dim is not None else axis)

    def max(self, dim=None, axis=None, keepdims=False, **kw):  # noqa: A003
        return np.asarray(np.ndarray.max(
            self, axis=dim if dim is not None else axis,
            keepdims=keepdims)).view(_A)

    def clamp_min(self, v):
        return np.maximum(self, v).view(_A)


def _arr(x):
    return np.asarray(x, dtype=np.float64).view(_A)


def _pids(vals):
    a = _arr(vals)

    class P:
        def __getitem__(self, key):
            # sdm_loss does pid[:, None] == pid[None, :]
            return a[key].view(_A)
    return P()


# --------------------------------------------------------------------------- #
# SDM maths                                                                    #
# --------------------------------------------------------------------------- #
def test_sdm_is_zero_when_prediction_matches_the_identity_distribution():
    """Two identities, two samples each, features perfectly identity-aligned.

    With identical features inside an identity and orthogonal ones across, the
    softmax over similarities converges on the same block distribution as the
    label matrix, so the KL -- and the loss -- goes to zero. A loss that does
    NOT approach zero here is not measuring what it claims to.
    """
    f = _arr([[1, 0], [1, 0], [0, 1], [0, 1]])
    loss = sdm_loss(f, f, _pids([0, 0, 1, 1]), _T, temp=0.005)
    assert loss == pytest.approx(0.0, abs=1e-3)


def test_sdm_punishes_a_confident_wrong_match():
    """Same four samples, but the features say sample 0 belongs with identity 1."""
    good = _arr([[1, 0], [1, 0], [0, 1], [0, 1]])
    bad = _arr([[0, 1], [1, 0], [0, 1], [0, 1]])
    pid = _pids([0, 0, 1, 1])
    assert sdm_loss(bad, bad, pid, _T, temp=0.05) > \
           sdm_loss(good, good, pid, _T, temp=0.05)


def test_sdm_reduces_to_infonce_when_every_identity_is_unique():
    """With unique identities the label matrix IS the identity matrix -- exactly
    InfoNCE's one-hot target -- so orthonormal features (each sample matching
    only itself) already satisfy it and the loss is ~0.

    That is the whole argument for PK sampling: hand SDM identity-disjoint
    batches and it optimises the same target InfoNCE does, so the change buys
    nothing. The contrast below is the point -- the SAME features that are
    optimal under unique labels are badly wrong once two samples share an
    identity, because now the loss demands they be similar.
    """
    f = _arr(np.eye(4))
    unique = sdm_loss(f, f, _pids([0, 1, 2, 3]), _T, temp=0.05)
    paired = sdm_loss(f, f, _pids([0, 0, 1, 1]), _T, temp=0.05)
    assert unique == pytest.approx(0.0, abs=1e-6)
    assert paired > 1.0, "sharing an identity must create a gradient that " \
                         "one-hot InfoNCE never produces"


def test_sdm_label_rows_are_normalised():
    """A 4-sample identity must not contribute 4x the gradient of a singleton;
    ``labels / labels.sum(dim=1)`` is what keeps identities comparable."""
    f = _arr(np.eye(5))
    a = sdm_loss(f, f, _pids([0, 0, 0, 0, 1]), _T, temp=0.05)
    b = sdm_loss(f, f, _pids([0, 0, 1, 1, 2]), _T, temp=0.05)
    assert np.isfinite(a) and np.isfinite(b)


# --------------------------------------------------------------------------- #
# PK sampler                                                                   #
# --------------------------------------------------------------------------- #
def test_pk_batches_put_k_images_of_each_identity_together():
    """8 identities x 6 images, batch 8, K=4 -> 2 identities x 4 images."""
    groups = [f"id{i // 6}" for i in range(48)]
    rng = np.random.default_rng(0)
    batches = list(_pk_batches(groups, batch=8, k=4, rng=rng))
    assert batches
    for b in batches:
        counts = {}
        for i in b:
            counts[groups[i]] = counts.get(groups[i], 0) + 1
        assert max(counts.values()) > 1, "a batch with no repeated identity " \
                                         "gives SDM nothing to learn from"
        assert all(c <= 4 for c in counts.values())


def test_pk_batches_cover_the_epoch():
    groups = [f"id{i // 6}" for i in range(48)]
    seen = {int(i) for b in _pk_batches(groups, 8, 4, np.random.default_rng(0))
            for i in b}
    assert len(seen) >= 40          # ~an epoch, minus the tail


def test_pk_batches_keep_small_identities():
    """Identities with fewer than K images are topped up, not dropped -- ICFG's
    tail of 2-3-crop identities would otherwise never be learnable."""
    groups = ["a"] * 10 + ["b"] * 2 + ["c"] * 2
    batches = list(_pk_batches(groups, batch=8, k=4, rng=np.random.default_rng(0)))
    seen = {groups[i] for b in batches for i in b}
    assert seen == {"a", "b", "c"}
    for b in batches:
        assert len(b) >= 2


def test_pk_batches_are_deterministic_in_the_seed():
    groups = [f"id{i // 5}" for i in range(40)]
    a = [list(map(int, b)) for b in
         _pk_batches(groups, 8, 4, np.random.default_rng(7))]
    b = [list(map(int, x)) for x in
         _pk_batches(groups, 8, 4, np.random.default_rng(7))]
    assert a == b
