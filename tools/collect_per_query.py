#!/usr/bin/env python3
"""Collect the per-query evaluation outputs behind every number in the paper
into compact, dataset-free .npz files (no captions, no image paths, no pixels).

This is a MAINTAINER tool: it reads the private results tree that the original
runs wrote (``--result-root``, e.g. ../Dataset/Result) and writes
``results/per_query/<model>__<benchmark>.npz`` plus ``results/registry.json``.
Users of the release never need to run it -- the .npz files are shipped.

Each .npz holds, for one (model, benchmark):
    cells      (C,)    cell names: "clean", "<geometry>_<NN>" (NN = 100*c), "oof_*"
    inp, ap    (C, Q)  per-query inverse negative penalty / average precision
    hit1/5/10  (C, Q)  per-query Rank-K hit (0/1)
Query order is the evaluation-set order (identical across cells and models).
"""
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np

GEOMS = ("torso", "cutout", "foliage")
LEVELS = (10, 20, 30, 40, 50)


def grid(clean, occ):
    """clean: path; occ: f(g, l) -> path  ->  ordered {cell: path}"""
    d = {"clean": clean}
    for g in GEOMS:
        for l in LEVELS:
            d[f"{g}_{l:02d}"] = occ(g, l)
    return d


def runs():
    R = {}
    # ---- published IRRA, scored by our harness (--encoder irra) -------------
    R[("irra_published", "cuhk")] = grid("gate/irra_published_cuhk.json",
        lambda g, l: f"occ_study/irra_published/occ/cuhk_{g}_l{l}")
    # RSTPReid / ICFG-PEDES grids of the published IRRA exist only as aggregate
    # rows (the per-query files were not kept); see export_aggregate() below.
    # ---- D (baseline, 60 ep) and D+A seed 0 ---------------------------------
    R[("D", "cuhk")] = grid("260915/isatmp_D_cuhk_author_sdm/eval_cuhk_gallery",
        lambda g, l: f"occ_study/occ/cuhk_{g}_l{l}_D")
    R[("DA_s0", "cuhk")] = grid("261002/drone/eval/Docc/cuhk_clean",
        lambda g, l: f"occ_study/occ/cuhk_{g}_l{l}_Docc")
    for m, run in (("D", "isatmp_D_cuhk_author_sdm"), ("DA_s0", "isatmp_Docc_cuhk_author_sdm")):
        for b in ("rstp", "icfg"):
            R[(m, b)] = grid(f"261016/{run}/{b}_clean",
                             lambda g, l, b=b, run=run: f"261016/{run}/occ/{b}_{g}_l{l}")
    # ---- revision runs (2026-10-17) ----------------------------------------
    new = {"DA_s1": ("Docc_seed1", ("cuhk", "rstp", "icfg")),   # icfg: optional, see OPTIONAL
           "DA_s2": ("Docc_seed2", ("cuhk", "rstp", "icfg")),
           "D_plus10": ("ctrl_D_plus10_noocc", ("cuhk", "rstp", "icfg")),
           "DA_prime": ("Docc_lv_025_055_07", ("cuhk", "rstp")),
           "irra_init": ("irra_cuhk_init", ("cuhk", "rstp", "icfg")),
           "irra_plus_A": ("irra_plus_occ", ("cuhk", "rstp", "icfg"))}
    for m, (run, benches) in new.items():
        for b in benches:
            R[(m, b)] = grid(f"261017/{run}/{b}_clean",
                             lambda g, l, b=b, run=run: f"261017/{run}/occ/{b}_{g}_l{l}")
    # ---- CLIP zero-shot (CUHK only, Table 1) --------------------------------
    R[("clip_zs", "cuhk")] = grid("261002/drone/eval/clip_zs/cuhk_clean",
        lambda g, l: f"occ_study/occ/cuhk_{g}_l{l}_clip_zs")
    # ---- out-of-family (OOF) occluders: real objects, MGCC area ratios ------
    for m, tag in (("D", "D"), ("DA_s0", "Docc")):
        R[(m, "cuhk")]["oof_mgcc_one_per_id"] = f"occ_study/occ/cuhk_ogor_l00_one_per_id_{tag}"
        for l in (25, 50, 70):
            R[(m, "rstp")][f"oof_sweep_{l}"] = f"occ_study/occ/rstp_ogor_l{l}_{tag}"
        R[(m, "rstp")]["oof_mgcc_one_per_id"] = f"occ_study/occ/rstp_ogor_l00_one_per_id_{tag}"
    # ---- aerial (NOMAD-val gallery) + ground cost ---------------------------
    R[("clip_zs", "nomad")] = {"clean": "261002/drone/eval/clip_zs/nomad_gallery_val"}
    R[("clip_zs", "rstp")] = {"clean": "261002/drone/eval/clip_zs/rstp_clean"}
    R[("DA_s0", "nomad")] = {"clean": "261002/drone/eval/Docc/nomad_gallery_val"}
    for s in range(3):
        base = f"261016/drone/eval/docc_nomad_last_s{s}"
        R[(f"DA_nomad_s{s}", "nomad")] = {"clean": f"{base}/nomad_gallery_val"}
        R[(f"DA_nomad_s{s}", "cuhk")] = {"clean": f"{base}/cuhk_clean"}
        R[(f"DA_nomad_s{s}", "rstp")] = {"clean": f"{base}/rstp_clean"}
    return R


#: (model, benchmark) pairs scored after the first release; skipped while missing.
#: D+A seeds 1-2 on ICFG-PEDES: see docs/STATISTICS.md, section 5
OPTIONAL = {("DA_s1", "icfg"), ("DA_s2", "icfg")}


def complete(root: Path, cells: dict) -> bool:
    for rel in cells.values():
        p = root / rel
        if not ((p / "metrics.json") if p.is_dir() else p).exists():
            return False
    return True


def load(root: Path, rel: str):
    p = root / rel
    if p.is_dir():
        p = p / "metrics.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    pq = d["per_query"]
    qi = np.asarray(pq["query_index"])
    assert (qi == np.arange(len(qi))).all(), f"query order not canonical: {p}"
    arr = {k: np.asarray(pq[k], np.float64) for k in ("inp", "ap", "hit@1", "hit@5", "hit@10")}
    ov = d["overall"]
    for k_pq, k_ov in (("inp", "mINP"), ("ap", "mAP"), ("hit@1", "R@1")):
        assert abs(arr[k_pq].mean() - ov[k_ov]) < 1e-9, (p, k_ov)
    return arr, d, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.result_root), Path(a.out)
    (out / "per_query").mkdir(parents=True, exist_ok=True)
    reg = {}
    for (m, b), cells in runs().items():
        if (m, b) in OPTIONAL and not complete(root, cells):
            print(f"{m:16s} {b:6s} not (fully) scored yet -- skipped")
            continue
        names, A = [], {k: [] for k in ("inp", "ap", "hit@1", "hit@5", "hit@10")}
        src, overall, ckpt, extra = {}, {}, None, {}
        nq = None
        for c, rel in cells.items():
            arr, d, p = load(root, rel)
            if nq is None:
                nq = len(arr["inp"])
            assert len(arr["inp"]) == nq, (m, b, c)
            names.append(c)
            for k in A:
                A[k].append(arr[k])
            src[c] = str(p.relative_to(root)).replace("\\", "/")
            overall[c] = {k: d["overall"][k] for k in ("R@1", "R@5", "R@10", "mAP", "mINP", "n_query", "n_gallery")}
            if c == "clean":
                argv = d.get("argv") or {}
                ckpt = argv.get("ckpt") or argv.get("irra_ckpt")
                extra = {"encoder": argv.get("encoder"), "img_size": argv.get("img_size"),
                         "benchmark_stats": d.get("stats"), "breakdown": d.get("breakdown")}
        np.savez_compressed(out / "per_query" / f"{m}__{b}.npz", cells=np.array(names),
                            inp=np.array(A["inp"], np.float32), ap=np.array(A["ap"], np.float32),
                            hit1=np.array(A["hit@1"], np.uint8), hit5=np.array(A["hit@5"], np.uint8),
                            hit10=np.array(A["hit@10"], np.uint8))
        reg[f"{m}__{b}"] = {"model": m, "benchmark": b, "n_query": nq, "cells": names,
                            "checkpoint_used": ckpt, "source_metrics": src, "overall": overall, **extra}
        print(f"{m:16s} {b:6s} cells={len(names):2d} nq={nq:6d}  clean R@1={100*overall['clean']['R@1']:.2f} mINP={100*overall['clean']['mINP']:.2f}")
    (out / "registry.json").write_text(json.dumps(reg, indent=1, default=str), encoding="utf-8")
    export_aggregate(root, out)


TABLE = "261016/tables/occlusion_table.csv"


def export_aggregate(root: Path, out: Path):
    """Aggregate-only cells (no per-query file): published IRRA on RSTPReid and
    ICFG-PEDES (Fig. 1). Copied verbatim from the original results table."""
    keep = []
    for r in csv.DictReader(open(root / TABLE, encoding="utf-8")):
        if r["model"] == "irra_published" and r["benchmark"] in ("rstp", "icfg") \
                and r["variant"] in ("clean",) + GEOMS:
            keep.append(r)
    cols = ["model", "benchmark", "variant", "level", "n_query", "n_gallery",
            "R@1", "R@5", "R@10", "mAP", "mINP"]
    with open(out / "aggregate_only_cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(keep)
    print(f"aggregate-only rows: {len(keep)} -> {out/'aggregate_only_cells.csv'}")


if __name__ == "__main__":
    main()
