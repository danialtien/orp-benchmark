#!/usr/bin/env python3
"""Compare YOUR evaluation of a checkpoint with the paper's, cell by cell.

After scoring a model with tools/run_config.py eval (outputs under
<out-root>/<model>/eval/<cell>/metrics.json), run

    python tools/compare_runs.py --out-root runs                 # every model found
    python tools/compare_runs.py --out-root runs --models DA_s0 D

For every (model, benchmark, cell) that exists both in your runs and in
results/per_query/, this prints the paper's and your Rank-1 / mAP / mINP and the
share of queries whose Rank-1 hit is identical. It also writes your per-query
results in the release format to <out-root>/results/per_query/, so that

    ORP_RESULTS=<out-root>/results python analysis/seed_consistency.py

re-runs the statistics on your own scores (once all the files it needs exist).

Expect tiny differences, not identity: GPU kernels and AMP make the embeddings
non-deterministic in the last bits, which can swap near-tied ranks.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "results"
GEOMS, LEVELS = ("torso", "cutout", "foliage"), (10, 20, 30, 40, 50)
#: run-folder name -> per-query file name, where they differ
ALIAS = {"IRRA_plus_A": "irra_plus_A"}


def cells_for(bench: str) -> dict[str, str]:
    """release cell name -> eval sub-folder written by tools/run_config.py"""
    if bench == "nomad":
        return {"clean": "nomad_val"}
    d = {"clean": f"{bench}_clean"}
    d.update({f"{g}_{l:02d}": f"{bench}_{g}_l{l:02d}" for g in GEOMS for l in LEVELS})
    d["oof_mgcc_one_per_id"] = f"{bench}_ogor_l00_one_per_id"
    if bench == "rstp":
        d.update({f"oof_sweep_{l}": f"rstp_ogor_l{l}" for l in (25, 50, 70)})
    return d


def load_metrics(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    pq = d["per_query"]
    qi = np.asarray(pq["query_index"])
    order = np.argsort(qi)
    return {k: np.asarray(pq[src], np.float64)[order]
            for k, src in (("inp", "inp"), ("ap", "ap"), ("hit1", "hit@1"), ("hit5", "hit@5"), ("hit10", "hit@10"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default=str(REPO / "runs"))
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    root = Path(a.out_root)
    dst = root / "results"
    (dst / "per_query").mkdir(parents=True, exist_ok=True)
    for f in ("query_identities.npz", "aggregate_only_cells.csv", "registry.json"):
        shutil.copy2(PAPER / f, dst / f)
    models = a.models or sorted(p.name for p in root.iterdir() if (p / "eval").is_dir())
    print(f"{'model':14s} {'bench':6s} {'cell':20s} {'R@1 paper/yours':>17s} {'mAP paper/yours':>17s} "
          f"{'mINP paper/yours':>17s} {'same top-1':>10s}")
    worst = 0.0
    for m in models:
        name = ALIAS.get(m, m)
        for b in ("cuhk", "rstp", "icfg", "nomad"):
            got, names = {}, []
            for cell, sub in cells_for(b).items():
                p = root / m / "eval" / sub / "metrics.json"
                if p.exists():
                    got[cell] = load_metrics(p)
                    names.append(cell)
            if not got:
                continue
            np.savez_compressed(dst / "per_query" / f"{name}__{b}.npz", cells=np.array(names),
                                **{k: np.array([got[c][k] for c in names], np.float32 if k in ("inp", "ap") else np.uint8)
                                   for k in ("inp", "ap", "hit1", "hit5", "hit10")})
            ref_p = PAPER / "per_query" / f"{name}__{b}.npz"
            if not ref_p.exists():
                print(f"{m:14s} {b:6s} (no paper reference for this model/benchmark)")
                continue
            z = np.load(ref_p)
            rc = [str(c) for c in z["cells"]]
            for cell in names:
                if cell not in rc:
                    continue
                i, y = rc.index(cell), got[cell]
                if len(y["inp"]) != z["inp"].shape[1]:
                    print(f"{m:14s} {b:6s} {cell:20s} query count differs ({len(y['inp'])} vs {z['inp'].shape[1]})")
                    continue
                r = [(100 * z[k][i].mean(), 100 * y[k].mean()) for k in ("hit1", "ap", "inp")]
                same = 100 * (z["hit1"][i] == y["hit1"]).mean()
                worst = max(worst, *[abs(p_ - q_) for p_, q_ in r])
                print(f"{m:14s} {b:6s} {cell:20s} " + " ".join(f"{p_:7.2f}/{q_:7.2f}  " for p_, q_ in r)
                      + f"{same:8.2f}%")
    print(f"\nlargest |paper - yours| over all compared cells: {worst:.2f} points"
          f"\nyour per-query files: {dst / 'per_query'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
