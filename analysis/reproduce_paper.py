#!/usr/bin/env python3
"""Recompute every number in the ORP paper from the released per-query files.

    python analysis/reproduce_paper.py                 # B = 2000, seed 0
    python analysis/reproduce_paper.py --B 10000 --seed 1

No dataset, checkpoint or GPU is needed: the inputs are results/per_query/*.npz,
results/query_identities.npz and results/aggregate_only_cells.csv.

Output: a check table on stdout (paper value | recomputed | status) and
results/reproduced/paper_numbers.json.

Tolerances. Point estimates are deterministic and must agree to the printed
precision (0.01 for two-decimal numbers, 0.1 for one-decimal ratios).
Bootstrap CI endpoints are Monte-Carlo estimates. The paper's intervals were
drawn by the original analysis script, which used ONE numpy generator
(default_rng(0)) shared sequentially by all comparisons; here every comparison
gets a fresh default_rng(seed). The method (paired identity bootstrap, B = 2000,
percentile interval) is identical, but the different random stream moves an
endpoint by a few hundredths, so endpoints are checked with a tolerance of 0.15
percentage points. analysis/ci_stability.py shows the size of this wobble.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from math import lgamma, exp, log
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orp  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = json.loads((HERE / "paper_values.json").read_text(encoding="utf-8"))
ROWS = []


def check(name, paper, ours, tol, fmt="{:.2f}"):
    ok = abs(paper - ours) <= tol + 1e-9
    ROWS.append((name, fmt.format(paper), fmt.format(ours), "OK" if ok else "DIFF"))
    return ok


def check_ci(name, paper, ours, tol_pt=0.006, tol_ci=0.15):
    check(name + " (delta)", paper["delta"], ours["delta"], tol_pt)
    lo, hi = ours["ci"]
    check(name + " (CI low)", paper["ci"][0], lo, tol_ci)
    check(name + " (CI high)", paper["ci"][1], hi, tol_ci)


def aggregate_table():
    """Published IRRA on RSTPReid / ICFG-PEDES: aggregate rows only."""
    T = {}
    for r in csv.DictReader(open(orp.RESULTS / "aggregate_only_cells.csv", encoding="utf-8")):
        lv = int(round(float(r["level"]) * 100))
        cell = "clean" if r["variant"] == "clean" else f"{r['variant']}_{lv:02d}"
        T[(r["benchmark"], cell)] = {k: float(r[k]) for k in ("R@1", "mAP", "mINP")}
    return T


def chance_minp(k_per_query: np.ndarray, N: int) -> float:
    """Exact expected INP of a uniformly random ranking: E[k / max position of k
    correct items among N]. P(max = m) = C(m-1, k-1) / C(N, k)."""
    out = {}
    for k in np.unique(k_per_query):
        k = int(k)
        m = np.arange(k, N + 1)
        lp = (np.array([lgamma(x) for x in m]) - lgamma(k) - np.array([lgamma(x - k + 1) for x in m])
              - (lgamma(N + 1) - lgamma(k + 1) - lgamma(N - k + 1)))
        out[k] = float((np.exp(lp) * k / m).sum())
    return float(np.mean([out[int(k)] for k in k_per_query]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(orp.RESULTS / "reproduced" / "paper_numbers.json"))
    a = ap.parse_args()
    B, S = a.B, a.seed
    res = {}

    # ---------------- Sec. 3.2 / Fig. 1: Rank-1 hides the collapse ----------
    T = aggregate_table()
    def ret_agg(b, cell, m):
        return T[(b, cell)][m] / T[(b, "clean")][m]
    r1 = 100 * ret_agg("icfg", "foliage_50", "R@1"); rm = 100 * ret_agg("icfg", "foliage_50", "mINP")
    check("Fig1 IRRA ICFG Ret_R@1 foliage c=0.5 (%)", EXP["fig1_irra_icfg_foliage50_ret_R1"], r1, 0.05, "{:.1f}")
    check("Fig1 IRRA ICFG Ret_mINP foliage c=0.5 (%)", EXP["fig1_irra_icfg_foliage50_ret_mINP"], rm, 0.05, "{:.1f}")
    check("Fig1 IRRA ICFG Div foliage c=0.5 (x)", EXP["fig1_irra_icfg_div_foliage50"], r1 / rm, 0.05, "{:.1f}")
    dt = ret_agg("icfg", "torso_50", "R@1") / ret_agg("icfg", "torso_50", "mINP")
    check("Fig1 IRRA ICFG Div torso c=0.5 (x)", EXP["fig1_irra_icfg_div_torso50"], dt, 0.05, "{:.1f}")
    n_div = sum(ret_agg("icfg", c, "R@1") > ret_agg("icfg", c, "mINP") for c in orp.GRID)
    check("IRRA ICFG cells with Ret_R@1 > Ret_mINP", EXP["irra_icfg_cells_R1ret_gt_mINPret"], n_div, 0, "{:.0f}")
    irra_cuhk = orp.load("irra_published", "cuhk")
    dcu = orp.divergence(irra_cuhk)["foliage_50"]
    check("Div IRRA CUHK foliage c=0.5 (x)", EXP["div_irra_cuhk_foliage50"], dcu, 0.05, "{:.1f}")
    drs = ret_agg("rstp", "foliage_50", "R@1") / ret_agg("rstp", "foliage_50", "mINP")
    check("Div IRRA RSTPReid foliage c=0.5 (x)", EXP["div_irra_rstp_foliage50"], drs, 0.05, "{:.1f}")
    _, k_icfg, N_icfg = orp.identities("icfg")
    floor = 100 * chance_minp(k_icfg, N_icfg)
    ratio = 100 * T[("icfg", "foliage_50")]["mINP"] / floor
    check("IRRA ICFG mINP foliage c=0.5 / chance", EXP["irra_icfg_mINP_over_chance_foliage50"], ratio, 0.05, "{:.1f}")
    res["sec3_2"] = {"ret_R1": r1, "ret_mINP": rm, "div_icfg_fol50": r1 / rm, "div_icfg_torso50": dt,
                     "div_cuhk_fol50": dcu, "div_rstp_fol50": drs, "chance_mINP_icfg": floor,
                     "mINP_over_chance": ratio, "cells_R1ret_gt_mINPret_icfg": int(n_div)}

    # ---------------- Table 1 (CUHK-PEDES) ---------------------------------
    t1 = {}
    for m, exp_row in EXP["table1"].items():
        r = orp.load(m, "cuhk")
        row = {"R@1": 100 * r.mean("clean", "R@1"), "mINP": 100 * r.mean("clean"), "AURC": orp.aurc(r),
               "fol10": 100 * r.mean("foliage_10"), "fol30": 100 * r.mean("foliage_30"),
               "fol50": 100 * r.mean("foliage_50")}
        t1[m] = row
        for k, v in exp_row.items():
            check(f"Table1 {m} {k}", v, row[k], 0.006)
    res["table1"] = t1

    D, DA, D10 = orp.load("D", "cuhk"), orp.load("DA_s0", "cuhk"), orp.load("D_plus10", "cuhk")
    bt = orp.paired_bootstrap(D, DA, B, S)
    check_ci("CUHK D+A - D mINP-AURC", EXP["cuhk_DA_minus_D_AURC"], bt["d_aurc"])
    n15 = orp.cells_better(D10, DA, "absolute")
    check("CUHK cells where D+A > D+10 (abs. mINP)", EXP["cuhk_DA_beats_D10_cells"], n15, 0, "{:.0f}")
    b10 = orp.paired_bootstrap(D10, DA, B, S)
    check_ci("CUHK D+A - D+10 clean R@1", EXP["cuhk_DA_minus_D10_cleanR1"], b10["d_clean_R1"])
    seeds = [orp.aurc(orp.load(f"DA_s{s}", "cuhk")) for s in range(3)]
    check("CUHK D+A AURC, mean of 3 seeds", EXP["cuhk_DA_3seeds_AURC_mean"], float(np.mean(seeds)), 0.006)
    check("CUHK D+A AURC, SD of 3 seeds", EXP["cuhk_DA_3seeds_AURC_sd"], float(np.std(seeds, ddof=1)), 0.006)
    IR, IRA = orp.load("irra_published", "cuhk"), orp.load("irra_plus_A", "cuhk")
    bi = orp.paired_bootstrap(IR, IRA, B, S)
    check_ci("CUHK IRRA+A - IRRA(published) mINP-AURC", EXP["cuhk_irraA_minus_irra_AURC"], bi["d_aurc"])
    check("CUHK IRRA+A - IRRA clean R@1", EXP["cuhk_irraA_minus_irra_cleanR1"], bi["d_clean_R1"]["delta"], 0.006)
    res["cuhk_tests"] = {"DA-D": bt, "DA-D10": b10, "IRRA+A-IRRA": bi, "seed_AURC": seeds,
                         "cells_DA_gt_D10_abs": n15}

    # ---------------- Sec. 3.4: transfer and OOF -----------------------------
    for b, key in (("rstp", "rstp_DA_minus_D_AURC"), ("icfg", "icfg_DA_minus_D_AURC")):
        Db, DAb = orp.load("D", b), orp.load("DA_s0", b)
        bb = orp.paired_bootstrap(Db, DAb, B, S)
        check_ci(f"{b.upper()} D+A - D mINP-AURC", EXP[key], bb["d_aurc"])
        nc = orp.cells_better(Db, DAb, "absolute")
        check(f"{b.upper()} cells where D+A > D (abs. mINP)", EXP[f"{b}_DA_beats_D_cells"], nc, 0, "{:.0f}")
        res[f"{b}_DA-D"] = bb
    for m in ("D", "DA_s0"):
        v = 100 * orp.load(m, "cuhk").mean("oof_mgcc_one_per_id")
        check(f"OOF CUHK (MGCC protocol) mINP {m}", EXP["oof_cuhk_mINP"][m], v, 0.006)
        v = 100 * orp.load(m, "rstp").mean("oof_sweep_50")
        check(f"OOF RSTPReid c=0.5 mINP {m}", EXP["oof_rstp50_mINP"][m], v, 0.006)
    t2 = {}
    for m, exp_row in EXP["table2"].items():
        rs, ic = orp.load(m, "rstp"), orp.load(m, "icfg")
        row = {"rstp_R@1": 100 * rs.mean("clean", "R@1"), "rstp_AURC": orp.aurc(rs),
               "rstp_fol50": 100 * rs.mean("foliage_50"), "icfg_AURC": orp.aurc(ic)}
        t2[m] = row
        for k, v in exp_row.items():
            check(f"Table2 {m} {k}", v, row[k], 0.006)
    res["table2"] = t2

    # ---------------- Sec. 3.5 / Table 3: aerial ----------------------------
    reg = json.loads((orp.RESULTS / "registry.json").read_text(encoding="utf-8"))
    def m30(key):
        return 100 * reg[key]["breakdown"]["altitude_m"]["30"]["mINP"]
    t3 = {}
    for m in ("clip_zs", "DA_s0"):
        n = orp.load(m, "nomad")
        t3[m] = {"mAP": 100 * n.mean("clean", "mAP"), "mINP": 100 * n.mean("clean"), "mINP_30m": m30(f"{m}__nomad"),
                 "cuhk_R@1": 100 * orp.load(m, "cuhk").mean("clean", "R@1"),
                 "rstp_R@1": 100 * orp.load(m, "rstp").mean("clean", "R@1")}
    per_seed = []
    for s in range(3):
        n = orp.load(f"DA_nomad_s{s}", "nomad")
        per_seed.append({"mAP": 100 * n.mean("clean", "mAP"), "mINP": 100 * n.mean("clean"),
                         "mINP_30m": m30(f"DA_nomad_s{s}__nomad"),
                         "cuhk_R@1": 100 * orp.load(f"DA_nomad_s{s}", "cuhk").mean("clean", "R@1"),
                         "rstp_R@1": 100 * orp.load(f"DA_nomad_s{s}", "rstp").mean("clean", "R@1")})
    t3["DA_nomad"] = {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}
    t3["DA_nomad_sd"] = {k: float(np.std([p[k] for p in per_seed], ddof=1)) for k in per_seed[0]}
    t3["DA_nomad_per_seed"] = per_seed
    for m in ("clip_zs", "DA_s0", "DA_nomad"):
        for k, v in EXP["table3"][m].items():
            check(f"Table3 {m} {k}", v, t3[m][k], 0.006)
    check("Table3 Stage-2 mINP SD over seeds", EXP["nomad_stage2_mINP_sd"], t3["DA_nomad_sd"]["mINP"], 0.006)
    base = orp.load("DA_s0", "nomad").cell("clean")
    s2 = np.mean([orp.load(f"DA_nomad_s{s}", "nomad").cell("clean") for s in range(3)], 0)
    check("NOMAD actors improved (mINP, seed-mean)", EXP["nomad_actors_improved"], int((s2 > base).sum()), 0, "{:.0f}")
    check("Stage-2 ground cost, CUHK R@1 points", EXP["nomad_ground_cost_cuhk_R1"],
          t3["DA_s0"]["cuhk_R@1"] - t3["DA_nomad"]["cuhk_R@1"], 0.006)
    t3["actor_bootstrap_10k"] = {"stage2_mINP_ci": orp.mean_bootstrap(100 * s2, 10000, S),
                                 "delta_mINP_ci": orp.mean_bootstrap(100 * (s2 - base), 10000, S)}
    res["table3"] = t3

    # ---------------- report -----------------------------------------------
    w = max(len(r[0]) for r in ROWS)
    print(f"{'quantity':<{w}}  {'paper':>8}  {'ours':>8}  status")
    print("-" * (w + 28))
    for n, p, o, st in ROWS:
        print(f"{n:<{w}}  {p:>8}  {o:>8}  {st}")
    n_bad = sum(r[3] != "OK" for r in ROWS)
    print("-" * (w + 28))
    print(f"{len(ROWS) - n_bad}/{len(ROWS)} checks OK  (bootstrap B={B}, seed={S})")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res["checks"] = [dict(zip(("quantity", "paper", "recomputed", "status"), r)) for r in ROWS]
    out.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
