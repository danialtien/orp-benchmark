#!/usr/bin/env python3
"""How stable are the reported 95% CIs, and why resample identities?

    python analysis/ci_stability.py

For each paired comparison reported with a CI in the paper this prints
  (a) the paired identity-bootstrap CI for 5 RNG seeds at B = 2000 and once at
      B = 10000 -> the Monte-Carlo wobble of the endpoints;
  (b) the same CI from a naive QUERY-level bootstrap, which ignores that the
      queries of one person share gallery images -> it is narrower, i.e.
      over-confident. The paper uses (a).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orp  # noqa: E402

PAIRS = [("cuhk", "D", "DA_s0", "d_aurc"), ("cuhk", "D_plus10", "DA_s0", "d_clean_R1"),
         ("cuhk", "irra_published", "irra_plus_A", "d_aurc"),
         ("rstp", "D", "DA_s0", "d_aurc"), ("icfg", "D", "DA_s0", "d_aurc")]


def query_level(a, b, key, B=2000, seed=0):
    """Same statistic, but every query is its own resampling unit."""
    n = len(a.cell("clean"))
    rng = np.random.default_rng(seed)
    W = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=B)]).astype(float)
    inv, n_id, den = np.arange(n), n, W.sum(1)
    sa, sb = (orp._boot_stats(r, inv, n_id, W, den) for r in (a, b))
    d = {"d_aurc": sb["aurc"] - sa["aurc"], "d_clean_R1": sb["clean_R1"] - sa["clean_R1"]}[key]
    return np.percentile(d[1:], [2.5, 97.5])


def main():
    for bench, A, Bm, key in PAIRS:
        a, b = orp.load(A, bench), orp.load(Bm, bench)
        print(f"\n{bench}: {Bm} - {A}  [{key}]")
        for s in range(5):
            r = orp.paired_bootstrap(a, b, 2000, s)[key]
            print(f"  identity bootstrap B=2000  seed {s}: {orp.fmt_ci(r)}")
        r = orp.paired_bootstrap(a, b, 10000, 123)[key]
        print(f"  identity bootstrap B=10000 seed 123: {orp.fmt_ci(r)}")
        lo, hi = query_level(a, b, key)
        print(f"  query-level bootstrap (NOT used)  : [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
