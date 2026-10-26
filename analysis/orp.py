"""Occlusion Retention Protocol (ORP): metrics, AURC and paired identity bootstrap.

Everything here works on the released per-query files in ``results/per_query``
(see ``results/README.md``), so every number in the paper can be recomputed
without the datasets, the checkpoints or a GPU.

Definitions (Sec. 2.1 of the paper)
-----------------------------------
For a query q with |G_q| correct gallery images, let R_q^hard be the rank of the
LAST correct image. Then

    INP_q  = |G_q| / R_q^hard              mINP = mean_q INP_q
    Ret_M(g, c) = M(g, c) / M(clean)       (retention of metric M)
    AURC_M = 100 * mean_g [ trapezoid of Ret_M(g, c) over c = 0.1 ... 0.5 ] / 0.4
    Div(g, c) = Ret_R@1(g, c) / Ret_mINP(g, c)

with geometries g in {torso, cutout, foliage} and nominal coverages
c in {0.1, 0.2, 0.3, 0.4, 0.5} (15 cells). The clean point is used only as
the denominator of Ret; it is not a node of the trapezoid.

Confidence intervals
--------------------
``paired_bootstrap`` implements the CIs reported in the paper: a 95% percentile
interval from a *paired, identity-level (cluster)* bootstrap.

* The resampling unit is the test IDENTITY, not the query: queries of the same
  person are correlated (they share gallery images), so resampling queries
  independently would understate the variance.
* Each replicate draws n_id identities with replacement (multinomial weights
  w_i), and every query inherits the weight of its identity.
* Every metric of a replicate is a weighted mean over queries, and AURC is
  recomputed from scratch inside the replicate (clean value included), i.e. the
  ratio Ret = M(cell)/M(clean) is a ratio of two weighted means.
* PAIRED: the same weights are applied to both models, so the interval is for
  the difference B - A on the same resampled test population.
* B = 2000 replicates; the interval is the 2.5th-97.5th percentile of the
  replicate differences; the point estimate is the full-sample difference.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

GEOMS = ("torso", "cutout", "foliage")
LEVELS = (10, 20, 30, 40, 50)
GRID = [f"{g}_{l:02d}" for g in GEOMS for l in LEVELS]
#: per-query results to analyse; set ORP_RESULTS to analyse your own scores
#: (tools/compare_runs.py writes them in this format)
RESULTS = Path(os.environ.get("ORP_RESULTS") or Path(__file__).resolve().parents[1] / "results")
METRIC_KEYS = {"mINP": "inp", "mAP": "ap", "R@1": "hit1", "R@5": "hit5", "R@10": "hit10"}


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    model: str
    benchmark: str
    cells: List[str]
    data: Dict[str, np.ndarray]          # key -> (C, Q) float64

    def cell(self, name: str, metric: str = "mINP") -> np.ndarray:
        return self.data[METRIC_KEYS[metric]][self.cells.index(name)]

    def mean(self, name: str, metric: str = "mINP") -> float:
        return float(self.cell(name, metric).mean())

    def has_grid(self) -> bool:
        return all(c in self.cells for c in ["clean"] + GRID)


def load(model: str, benchmark: str, root: Path = RESULTS) -> Run:
    z = np.load(root / "per_query" / f"{model}__{benchmark}.npz")
    data = {k: z[k].astype(np.float64) for k in ("inp", "ap", "hit1", "hit5", "hit10")}
    return Run(model, benchmark, [str(c) for c in z["cells"]], data)


def identities(benchmark: str, root: Path = RESULTS):
    z = np.load(root / "query_identities.npz")
    qid = z[f"{benchmark}_qid"]
    k = z[f"{benchmark}_k"] if f"{benchmark}_k" in z else None
    return qid, k, int(z[f"{benchmark}_N"])


# --------------------------------------------------------------------------- #
# point estimates                                                              #
# --------------------------------------------------------------------------- #
def retention(run: Run, metric: str = "mINP") -> Dict[str, float]:
    c0 = run.mean("clean", metric)
    return {c: run.mean(c, metric) / c0 for c in GRID}


def aurc_from_ret(ret: Dict[str, float]) -> float:
    tot = 0.0
    for g in GEOMS:
        y = [ret[f"{g}_{l:02d}"] for l in LEVELS]
        tot += (0.5 * y[0] + sum(y[1:-1]) + 0.5 * y[-1]) / (len(y) - 1)
    return 100.0 * tot / len(GEOMS)


def aurc(run: Run, metric: str = "mINP") -> float:
    return aurc_from_ret(retention(run, metric))


def divergence(run: Run) -> Dict[str, float]:
    r1, rm = retention(run, "R@1"), retention(run, "mINP")
    return {c: r1[c] / rm[c] for c in GRID}


def cells_better(a: Run, b: Run, mode: str = "absolute", metric: str = "mINP") -> int:
    """Number of the 15 ORP cells where model ``b`` beats model ``a``.
    mode='absolute' compares M(cell); mode='retention' compares Ret_M(cell)."""
    if mode == "absolute":
        return sum(b.mean(c, metric) > a.mean(c, metric) for c in GRID)
    ra, rb = retention(a, metric), retention(b, metric)
    return sum(rb[c] > ra[c] for c in GRID)


# --------------------------------------------------------------------------- #
# paired identity-level bootstrap                                              #
# --------------------------------------------------------------------------- #
def _weights(benchmark: str, B: int, seed: int, root: Path = RESULTS):
    qid, _, _ = identities(benchmark, root)
    ids, inv = np.unique(qid, return_inverse=True)
    n_id = len(ids)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n_id, np.full(n_id, 1.0 / n_id), size=B).astype(np.float64)
    W = np.vstack([np.ones(n_id), W])          # row 0 = full sample (point estimate)
    counts = np.bincount(inv, minlength=n_id).astype(np.float64)
    return inv, n_id, W, W @ counts             # den: weighted number of queries


def _wmean(x: np.ndarray, inv, n_id, W, den) -> np.ndarray:
    """Weighted mean of a per-query vector x for every replicate -> (B+1,)."""
    return (W @ np.bincount(inv, weights=x, minlength=n_id)) / den


def _boot_stats(run: Run, inv, n_id, W, den, metric: str = "mINP"):
    """Per replicate: AURC of `metric`, clean value of `metric`, clean R@1,
    and the 15 absolute cell values -> dict of (B+1,) / (B+1, 15) arrays."""
    clean = _wmean(run.cell("clean", metric), inv, n_id, W, den)
    cells = np.stack([_wmean(run.cell(c, metric), inv, n_id, W, den) for c in GRID], 1)
    r = (cells / clean[:, None]).reshape(len(W), len(GEOMS), len(LEVELS))
    a = 100 * ((0.5 * r[:, :, 0] + r[:, :, 1:-1].sum(2) + 0.5 * r[:, :, -1]) / (len(LEVELS) - 1)).mean(1)
    r1 = _wmean(run.cell("clean", "R@1"), inv, n_id, W, den)
    return {"aurc": a, "clean": 100 * clean, "clean_R1": 100 * r1, "cells": 100 * cells}


def paired_bootstrap(a: Run, b: Run, B: int = 2000, seed: int = 0,
                     metric: str = "mINP", root: Path = RESULTS) -> dict:
    """95% paired identity-bootstrap CIs for (b - a): AURC, clean metric, clean R@1,
    and per-cell absolute differences."""
    assert a.benchmark == b.benchmark
    inv, n_id, W, den = _weights(a.benchmark, B, seed, root)
    sa = _boot_stats(a, inv, n_id, W, den, metric)
    sb = _boot_stats(b, inv, n_id, W, den, metric)

    def summ(d):
        lo, hi = np.percentile(d[1:], [2.5, 97.5], axis=0)
        return {"delta": d[0], "ci": (lo, hi), "sd": d[1:].std(ddof=1),
                "p_le_0": float((d[1:] <= 0).mean())}

    out = {"A": a.model, "B": b.model, "benchmark": a.benchmark, "n_identities": n_id,
           "B_replicates": B, "seed": seed,
           "aurc_A": sa["aurc"][0], "aurc_B": sb["aurc"][0],
           "d_aurc": summ(sb["aurc"] - sa["aurc"]),
           "d_clean": summ(sb["clean"] - sa["clean"]),
           "d_clean_R1": summ(sb["clean_R1"] - sa["clean_R1"])}
    dc = sb["cells"] - sa["cells"]
    lo, hi = np.percentile(dc[1:], [2.5, 97.5], axis=0)
    out["cells_point_positive"] = int((dc[0] > 0).sum())
    out["cells_ci_above_0"] = int((lo > 0).sum())
    out["cells_ci_below_0"] = int((hi < 0).sum())
    return out


def mean_bootstrap(x: np.ndarray, B: int = 10000, seed: int = 0):
    """Plain percentile bootstrap of a mean over independent units (used for the
    20 NOMAD-val actors, where each query is one actor)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    return tuple(np.percentile(x[idx].mean(1), [2.5, 97.5]))


def fmt_ci(d: dict, nd: int = 2) -> str:
    lo, hi = d["ci"]
    return f"{d['delta']:+.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
