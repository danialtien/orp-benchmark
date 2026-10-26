"""Collect ``metrics.json`` files into the tables the paper reports.

Reads a results tree written by ``eval_protocol`` (and optionally ``efficiency``)
and emits Markdown + CSV:

  T1  standard benchmarks   : one row per model, per benchmark
  T2  parameter efficiency  : params / VRAM / throughput / R@1-per-M-param
  T3  occlusion robustness  : one row per (variant, level), incl. RRR and dmINP
  T4/T5 aerial breakdown    : NOMAD by visibility / altitude
  SIG paired bootstrap      : is model A really better than model B?

    python -m sar_person.report_tables --results ../Dataset/Result \
        --out ../Dataset/Result/tables

    # significance between two runs on the same benchmark
    python -m sar_person.report_tables --compare \
        ../Dataset/Result/tier1/icfg_lora/metrics.json \
        ../Dataset/Result/tier1/icfg_zeroshot/metrics.json --metric ap
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .eval_protocol import paired_bootstrap

log = logging.getLogger(__name__)

# Rsum last: it is carried for comparability with MGCC's tables, not because it
# adds signal over the three R@K it sums.
METRIC_COLS = ["R@1", "R@5", "R@10", "mAP", "mINP", "Rsum"]

#: occlude_bench names its manifests "<benchmark>_<variant>_l<NN>"
_DERIVED = re.compile(r"^(?P<base>.+)_(?P<variant>[a-z]+)_l(?P<level>\d{2,3})$")


def _split_derived(name: str):
    """('rstp_torso_l70') -> ('rstp', 'torso', 0.70); None for a clean benchmark."""
    m = _DERIVED.match(name)
    if not m:
        return None
    return m["base"], m["variant"], int(m["level"]) / 100


def _load_all(root: Path) -> List[Dict]:
    out = []
    for p in sorted(root.rglob("metrics.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "overall" not in d:               # train_dataset's metrics.json
            continue
        d["_path"] = str(p)
        d["_run"] = p.parent.name
        out.append(d)
    log.info("[report] %d eval run(s) under %s", len(out), root)
    return out


def _pct(v):
    return "–" if v is None else f"{100 * v:.2f}"


def _md(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([line, sep, *body])


# --------------------------------------------------------------------------- #
# Tables                                                                       #
# --------------------------------------------------------------------------- #
def table_standard(runs: List[Dict]) -> str:
    """T1 -- one row per (model, benchmark) for the non-derived benchmarks."""
    rows = []
    for r in runs:
        bm = r["benchmark"]
        if _split_derived(bm):          # occluded variants belong in T3, not T1
            continue
        o = r["overall"]
        rows.append([r.get("model", r["_run"]), bm, o["n_gallery"], o["n_query"],
                     *[_pct(o.get(m)) for m in METRIC_COLS]])
    rows.sort(key=lambda x: (x[1], x[0]))
    return "## T1 — Standard benchmarks (t2i, %)\n\n" + _md(
        ["Model", "Benchmark", "|G|", "|Q|", *METRIC_COLS], rows)


def table_occlusion(runs: List[Dict]) -> str:
    """T3 -- occluded variants, with retention relative to the clean run.

    RRR (= R@1_occ / R@1_clean) is what makes methods with different clean
    baselines comparable; dmINP is the signal the occlusion-aware claim rests on.
    """
    clean: Dict[tuple, Dict] = {}
    derived: List[Dict] = []
    for r in runs:
        parsed = _split_derived(r["benchmark"])
        if parsed:
            derived.append({**r, "_v": parsed})
        else:
            clean[(r.get("model"), r["benchmark"])] = r

    rows = []
    for r in derived:
        base_bm, variant, level = r["_v"]
        c = clean.get((r.get("model"), base_bm))
        o, co = r["overall"], (c or {}).get("overall", {})
        rrr = (o["R@1"] / co["R@1"]) if co.get("R@1") else None
        dminp = (co["mINP"] - o["mINP"]) if co.get("mINP") is not None else None
        rows.append([r.get("model", r["_run"]), base_bm, variant,
                     f"{level:.2f}", *[_pct(o.get(m)) for m in METRIC_COLS],
                     "–" if rrr is None else f"{100*rrr:.1f}",
                     "–" if dminp is None else f"{100*dminp:+.2f}"])
    rows.sort(key=lambda x: (x[1], x[0], x[2], x[3]))
    note = ("\n\n*RRR = R@1 occluded / R@1 clean (higher = more robust). "
            "ΔmINP = clean − occluded (lower = more robust). "
            "A blank means the matching clean run was not found in this tree.*")
    return "## T3 — Occlusion robustness\n\n" + _md(
        ["Model", "Benchmark", "Variant", "Level", *METRIC_COLS, "RRR", "ΔmINP"],
        rows) + note


def table_breakdown(runs: List[Dict]) -> str:
    """T4/T5 -- per-bucket tables (NOMAD visibility / altitude)."""
    out = []
    for r in runs:
        for fld, buckets in (r.get("breakdown") or {}).items():
            rows = [[label, m["n_gallery_in_bucket"], m.get("n_query", "–"),
                     *[_pct(m.get(k)) for k in METRIC_COLS]]
                    for label, m in buckets.items()]
            out.append(f"### {r['benchmark']} · {r.get('model', r['_run'])} · "
                       f"by `{fld}`\n\n" +
                       _md(["Bucket", "|G|", "|Q|", *METRIC_COLS], rows))
    if not out:
        return ""
    return "## T4/T5 — Breakdowns\n\n" + "\n\n".join(out)


def table_efficiency(root: Path) -> str:
    """T2 -- from efficiency.py outputs anywhere under the results tree."""
    rows = []
    for p in sorted(root.rglob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(d, dict) or "trainable_params" not in d:
            continue
        rows.append([
            p.stem, d.get("mode", "?"), d.get("lora_rank", "–"),
            f"{d['trainable_params']/1e6:.2f}",
            f"{d['trainable_pct']:.2f}",
            f"{d['total_params']/1e6:.1f}",
            d.get("peak_vram_mb", "–"), d.get("images_per_sec", "–"),
            d.get("texts_per_sec", "–"),
        ])
    if not rows:
        return ""
    return ("## T2 — Parameter efficiency\n\n" + _md(
        ["Run", "Mode", "rank", "Trainable (M)", "Trainable %", "Total (M)",
         "Peak VRAM (MB)", "img/s", "txt/s"], rows) +
        "\n\n*Fill the `R@1 per M trainable param` column by dividing the T1 "
        "R@1 by `Trainable (M)` for the matching run.*")


# --------------------------------------------------------------------------- #
# Significance                                                                 #
# --------------------------------------------------------------------------- #
def compare(path_a: str, path_b: str, metric: str = "ap", seed: int = 0) -> Dict:
    """Paired query-level bootstrap between two eval runs.

    Both runs must be on the same benchmark: the per-query vectors are paired by
    position, and ``eval_protocol`` stores the query index it kept, so a
    mismatch is detected rather than silently producing nonsense.
    """
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))
    na, nb = a["benchmark"], b["benchmark"]
    # clean-vs-occluded IS a valid paired comparison (same queries, same pids,
    # only the gallery images were degraded), so allow a derived variant against
    # its own base; anything else is comparing apples to oranges.
    bases = {_split_derived(n)[0] if _split_derived(n) else n for n in (na, nb)}
    if na != nb and len(bases) > 1:
        raise SystemExit(f"different benchmarks: {na} vs {nb}")
    if na != nb:
        log.info("[compare] paired across variants of %s: %s vs %s",
                 bases.pop(), na, nb)
    ia = np.asarray(a["per_query"]["query_index"])
    ib = np.asarray(b["per_query"]["query_index"])
    if not np.array_equal(ia, ib):
        raise SystemExit("the two runs kept different query sets; rebuild both "
                         "from the same manifest before comparing")
    va = np.asarray(a["per_query"][metric], float)
    vb = np.asarray(b["per_query"][metric], float)
    res = paired_bootstrap(va, vb, seed=seed)
    res.update({"benchmark": a["benchmark"], "metric": metric,
                "model_a": a.get("model"), "model_b": b.get("model"),
                "mean_a": float(va.mean()), "mean_b": float(vb.mean())})
    return res


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=None, help="root of the results tree")
    ap.add_argument("--out", default=None, help="dir for tables.md + *.csv")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None)
    ap.add_argument("--metric", default="ap",
                    help="per-query vector for --compare: ap | inp | hit@1 ...")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.compare:
        print(json.dumps(compare(*args.compare, metric=args.metric,
                                 seed=args.seed), indent=2))
        return
    if not args.results:
        raise SystemExit("pass --results (or --compare A B)")

    root = Path(args.results)
    runs = _load_all(root)
    if not runs:
        raise SystemExit(f"no eval metrics.json under {root}")

    blocks = [t for t in (table_standard(runs), table_efficiency(root),
                          table_occlusion(runs), table_breakdown(runs)) if t]
    md = "# Results\n\n" + "\n\n".join(blocks) + "\n"
    print(md)

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tables.md").write_text(md, encoding="utf-8")
        with (out / "all_runs.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", "model", "benchmark", "n_gallery", "n_query",
                        *METRIC_COLS, "path"])
            for r in runs:
                o = r["overall"]
                w.writerow([r["_run"], r.get("model"), r["benchmark"],
                            o["n_gallery"], o["n_query"],
                            *[o.get(m) for m in METRIC_COLS], r["_path"]])
        print(f"-> {out/'tables.md'}\n-> {out/'all_runs.csv'}")


if __name__ == "__main__":
    main()
