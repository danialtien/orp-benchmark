#!/usr/bin/env python3
"""Check that the paper's main findings hold for EACH of the three training seeds.

    python analysis/seed_consistency.py            # writes results/reproduced/seed_consistency.{md,json}

Seeds. D+A (CUHK-PEDES, occlusion-aware fine-tuning from D) was trained with
--seed 0, 1, 2; the aerial Stage 2 (D+A -> NOMAD) also with seeds 0, 1, 2. The
seed controls the occlusion draws, data order, dropout, and the 300 identities
held out from the training split for checkpoint selection. D, D+10, D+A' and
IRRA+A are single-seed (seed 0) runs and serve as the fixed references.

Coverage. Seeds 1 and 2 of D+A were scored on CUHK-PEDES and RSTPReid (all 15
ORP cells + clean); at release time ICFG-PEDES was scored for seed 0 only. As
soon as results/per_query/DA_s1__icfg.npz and DA_s2__icfg.npz exist (see
docs/STATISTICS.md, "Pending"), ICFG-PEDES is added to every check below.

Findings checked per seed (paper section in brackets):
  F1 [3.2]  Rank-1 retention exceeds mINP retention in every ORP cell (Div > 1).
  F2 [3.3]  D+A raises mINP-AURC over D; paired identity-bootstrap CI excludes 0.
  F3 [3.3]  The gain is not explained by extra epochs: D+A beats D+10 in mINP-AURC
            (CI excludes 0) and in all 15 cells.
  F4 [3.3]  ...at no cost in clean Rank-1 relative to D+10 (CI includes 0 or is > 0).
  F5 [3.4]  The gain transfers to RSTPReid (unseen during training).
  F6 [3.5]  Stage 2 roughly triples NOMAD-val mINP over D+A for every seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orp  # noqa: E402


def per_seed(bench, s, B, seed):
    da = orp.load(f"DA_s{s}", bench)
    D, D10 = orp.load("D", bench), orp.load("D_plus10", bench)
    vD = orp.paired_bootstrap(D, da, B, seed)
    v10 = orp.paired_bootstrap(D10, da, B, seed)
    div = orp.divergence(da)
    return {
        "clean_R1": 100 * da.mean("clean", "R@1"), "clean_mINP": 100 * da.mean("clean"),
        "AURC": orp.aurc(da), "AURC_R1": orp.aurc(da, "R@1"),
        "fol50_mINP": 100 * da.mean("foliage_50"),
        "F1_cells_div_gt_1": sum(v > 1 for v in div.values()), "F1_min_div": min(div.values()),
        "F1_div_fol50": div["foliage_50"],
        "F2_dAURC_vs_D": vD["d_aurc"]["delta"], "F2_ci": vD["d_aurc"]["ci"],
        "F2_cells_abs_gt_D": orp.cells_better(D, da, "absolute"),
        "F2_cells_ret_gt_D": orp.cells_better(D, da, "retention"),
        "F3_dAURC_vs_D10": v10["d_aurc"]["delta"], "F3_ci": v10["d_aurc"]["ci"],
        "F3_cells_abs_gt_D10": orp.cells_better(D10, da, "absolute"),
        "F3_cells_ret_gt_D10": orp.cells_better(D10, da, "retention"),
        "F3_cells_ci_above_0_vs_D10": v10["cells_ci_above_0"],
        "F4_dR1_vs_D10": v10["d_clean_R1"]["delta"], "F4_ci": v10["d_clean_R1"]["ci"],
        "dR1_vs_D": vD["d_clean_R1"]["delta"], "dR1_vs_D_ci": vD["d_clean_R1"]["ci"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    ap.add_argument("--out", default=str(orp.RESULTS / "reproduced" / "seed_consistency"))
    a = ap.parse_args()
    PQ = orp.RESULTS / "per_query"
    benches = [b for b in ("cuhk", "rstp", "icfg")
               if all((PQ / f"DA_s{s}__{b}.npz").exists() for s in range(3))]
    R = {b: {s: per_seed(b, s, a.B, a.seed) for s in range(3)} for b in benches}
    ref = {b: {"D": orp.aurc(orp.load("D", b)), "D_plus10": orp.aurc(orp.load("D_plus10", b))}
           for b in benches}
    # cells where a seed of D+A does NOT beat D in absolute mINP (reported, not hidden)
    exceptions = {b: {s: [(c, round(100 * orp.load(f"DA_s{s}", b).mean(c), 2), round(100 * orp.load("D", b).mean(c), 2))
                          for c in orp.GRID if orp.load(f"DA_s{s}", b).mean(c) <= orp.load("D", b).mean(c)]
                      for s in range(3)} for b in benches}

    # aerial Stage 2, per seed
    base = orp.load("DA_s0", "nomad")
    aer = {}
    for s in range(3):
        n = orp.load(f"DA_nomad_s{s}", "nomad")
        d = n.cell("clean") - base.cell("clean")
        aer[s] = {"mAP": 100 * n.mean("clean", "mAP"), "mINP": 100 * n.mean("clean"),
                  "ratio_vs_DA": n.mean("clean") / base.mean("clean"),
                  "actors_improved": int((d > 0).sum()), "n_actors": int(d.size),
                  "dmINP": 100 * d.mean(), "dmINP_ci": orp.mean_bootstrap(100 * d, 10000, a.seed),
                  "cuhk_R1": 100 * orp.load(f"DA_nomad_s{s}", "cuhk").mean("clean", "R@1")}
    base_mINP = 100 * base.mean("clean")

    # ---------------- verdicts -------------------------------------------------
    V = []
    def verdict(name, ok, detail):
        V.append((name, "holds for all 3 seeds" if ok else "DOES NOT hold for every seed", detail))
    NAMES = {"cuhk": "CUHK-PEDES", "rstp": "RSTPReid", "icfg": "ICFG-PEDES"}
    for b in benches:
        nm, r = NAMES[b], R[b]
        verdict(f"F1 Div>1 in all 15 cells ({nm})", all(r[s]["F1_cells_div_gt_1"] == 15 for s in r),
                "min Div over cells per seed: " + ", ".join(f"{r[s]['F1_min_div']:.3f}" for s in r))
        verdict(f"F2 D+A > D, CI excludes 0 ({nm})", all(r[s]["F2_ci"][0] > 0 for s in r),
                "; ".join(f"s{s} {r[s]['F2_dAURC_vs_D']:+.2f} [{r[s]['F2_ci'][0]:.2f}, {r[s]['F2_ci'][1]:.2f}]" for s in r))
        verdict(f"F3 D+A > D+10, CI excludes 0 ({nm})", all(r[s]["F3_ci"][0] > 0 for s in r),
                "; ".join(f"s{s} {r[s]['F3_dAURC_vs_D10']:+.2f} [{r[s]['F3_ci'][0]:.2f}, {r[s]['F3_ci'][1]:.2f}]" for s in r))
        verdict(f"F3 D+A > D+10 in all 15 cells, absolute mINP ({nm})",
                all(r[s]["F3_cells_abs_gt_D10"] == 15 for s in r),
                "cells per seed: " + ", ".join(str(r[s]["F3_cells_abs_gt_D10"]) for s in r))
        verdict(f"F4 clean R@1 not below D+10 (CI upper > 0) ({nm})", all(r[s]["F4_ci"][1] > 0 for s in r),
                "; ".join(f"s{s} {r[s]['F4_dR1_vs_D10']:+.2f} [{r[s]['F4_ci'][0]:.2f}, {r[s]['F4_ci'][1]:.2f}]" for s in r))
    for b in [x for x in benches if x != "cuhk"]:
        r = R[b]
        verdict(f"F5 transfer: D+A > D in all 15 cells, mINP retention ({NAMES[b]})",
                all(r[s]["F2_cells_ret_gt_D"] == 15 for s in r),
                "cells per seed: " + ", ".join(str(r[s]["F2_cells_ret_gt_D"]) for s in r))
        verdict(f"F5 transfer: D+A > D in all 15 cells, absolute mINP ({NAMES[b]})",
                all(r[s]["F2_cells_abs_gt_D"] == 15 for s in r),
                "cells per seed: " + ", ".join(str(r[s]["F2_cells_abs_gt_D"]) for s in r)
                + "; exceptions (cell: D+A vs D mINP) " + ("; ".join(
                    f"s{s} {c} {x:.2f} vs {y:.2f}" for s in r for c, x, y in exceptions[b][s]) or "none"))
    verdict("F6 Stage 2 >= 2.5x D+A NOMAD-val mINP", all(aer[s]["ratio_vs_DA"] >= 2.5 for s in aer),
            "ratio per seed: " + ", ".join(f"{aer[s]['ratio_vs_DA']:.2f}x" for s in aer))

    # ---------------- markdown -------------------------------------------------
    L = ["# Seed consistency of the main findings", "",
         f"Generated by `analysis/seed_consistency.py` (paired identity bootstrap, B = {a.B}, RNG seed {a.seed}; "
         "NOMAD: actor bootstrap, B = 10,000). Values in %; CIs are 95%.", "",
         "## Verdicts", "", "| finding | result | detail |", "|---|---|---|"]
    L += [f"| {n} | {v} | {d} |" for n, v, d in V]
    TITLES = {"cuhk": "CUHK-PEDES (training domain)", "rstp": "RSTPReid (unseen)", "icfg": "ICFG-PEDES (unseen)"}
    for b in benches:
        nm, r = TITLES[b], R[b]
        L += ["", f"## {nm}", "",
              f"References (seed 0, single run): D mINP-AURC {ref[b]['D']:.2f}, D+10 {ref[b]['D_plus10']:.2f}.", "",
              "| quantity | seed 0 | seed 1 | seed 2 | mean ± SD |", "|---|---|---|---|---|"]
        def row(label, key, nd=2):
            xs = [r[s][key] for s in range(3)]
            return f"| {label} | " + " | ".join(f"{x:.{nd}f}" for x in xs) + f" | {np.mean(xs):.{nd}f} ± {np.std(xs, ddof=1):.{nd}f} |"
        def row_ci(label, key, ci):
            return f"| {label} | " + " | ".join(f"{r[s][key]:+.2f} [{r[s][ci][0]:.2f}, {r[s][ci][1]:.2f}]" for s in range(3)) + \
                   f" | {np.mean([r[s][key] for s in range(3)]):+.2f} ± {np.std([r[s][key] for s in range(3)], ddof=1):.2f} |"
        def row_int(label, key):
            return f"| {label} | " + " | ".join(str(r[s][key]) for s in range(3)) + " | – |"
        L += [row("clean R@1", "clean_R1"), row("clean mINP", "clean_mINP"), row("mINP-AURC", "AURC"),
              row("R@1-AURC", "AURC_R1"), row("foliage c=0.5 mINP", "fol50_mINP"),
              row_int("F1 cells with Div > 1 (of 15)", "F1_cells_div_gt_1"), row("F1 min Div", "F1_min_div", 3),
              row("F1 Div, foliage c=0.5", "F1_div_fol50", 3),
              row_ci("F2 ΔAURC vs D [CI]", "F2_dAURC_vs_D", "F2_ci"),
              row_int("F2 cells > D (absolute mINP)", "F2_cells_abs_gt_D"),
              row_int("F2 cells > D (mINP retention)", "F2_cells_ret_gt_D"),
              row_ci("F3 ΔAURC vs D+10 [CI]", "F3_dAURC_vs_D10", "F3_ci"),
              row_int("F3 cells > D+10 (absolute mINP)", "F3_cells_abs_gt_D10"),
              row_int("F3 cells > D+10 (mINP retention)", "F3_cells_ret_gt_D10"),
              row_int("F3 cells whose CI vs D+10 excludes 0", "F3_cells_ci_above_0_vs_D10"),
              row_ci("F4 Δ clean R@1 vs D+10 [CI]", "F4_dR1_vs_D10", "F4_ci"),
              row_ci("Δ clean R@1 vs D [CI]", "dR1_vs_D", "dR1_vs_D_ci")]
        ex = "; ".join(f"seed {s}: {c} ({x:.2f} vs {y:.2f})" for s in range(3) for c, x, y in exceptions[b][s])
        L += ["", "Cells where D+A does not beat D in absolute mINP (D+A vs D): " + (ex or "none") + "."]
    L += ["", "## NOMAD-val (aerial Stage 2, 20 held-out actors)", "",
          f"Reference: D+A (seed 0) mINP {base_mINP:.2f}.", "",
          "| quantity | seed 0 | seed 1 | seed 2 | mean ± SD |", "|---|---|---|---|---|"]
    for k, lab in (("mAP", "mAP"), ("mINP", "mINP"), ("ratio_vs_DA", "mINP ratio vs D+A"), ("cuhk_R1", "CUHK clean R@1")):
        xs = [aer[s][k] for s in range(3)]
        L.append(f"| {lab} | " + " | ".join(f"{x:.2f}" for x in xs) + f" | {np.mean(xs):.2f} ± {np.std(xs, ddof=1):.2f} |")
    L.append("| Δ mINP vs D+A [actor CI] | " + " | ".join(
        f"{aer[s]['dmINP']:+.2f} [{aer[s]['dmINP_ci'][0]:.2f}, {aer[s]['dmINP_ci'][1]:.2f}]" for s in range(3)) + " | – |")
    L.append("| actors improved | " + " | ".join(f"{aer[s]['actors_improved']}/{aer[s]['n_actors']}" for s in range(3)) + " | – |")
    L += ["", "## Not covered by seeds", ""]
    if "icfg" not in benches:
        L.append("* ICFG-PEDES transfer was scored for seed 0 only (seeds 1-2: CUHK-PEDES and RSTPReid); "
                 "pending runs are listed in docs/STATISTICS.md.")
    L += [
          "* D, D+10, D+A' and IRRA+A are single-seed runs; seed variation of the references is not estimated.",
          "* The paper's Rank-1-vs-mINP divergence on the published IRRA (Fig. 1) involves no training and no seed."]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"per_seed": R, "references": ref, "aerial": aer,
                                                    "exceptions_vs_D": exceptions,
                                                    "verdicts": V}, indent=1, default=float), encoding="utf-8")
    print("\n".join(L))
    n_fail = sum(not v[1].startswith("holds") for v in V)
    print(f"\n{len(V) - n_fail}/{len(V)} findings hold for all three seeds "
          f"-> {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
