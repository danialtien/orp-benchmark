"""Turn one ``metrics.json`` into a comparison table + chart against prior work.

    python -m sar_person.compare_related --metrics .../metrics.json
    python -m sar_person.compare_related --results ../Dataset/Result --out .../cmp

Writes next to the metrics file (or to ``--out``):

    comparison.md    ranked table, your run highlighted, sources in the footer
    comparison.csv   the same, for a spreadsheet
    comparison.png   grouped bar chart over R@1 / R@5 / R@10 / mAP / mINP

Only rows measured on **the same benchmark** are ever put in one table. For a
benchmark with no published protocol (nomad, forestperson, cuhk_heldout) the
tool writes the table with your runs alone plus a note saying why -- comparing
a NOMAD R@1 against a CUHK table would be the single easiest way to produce a
figure that is both impressive and meaningless.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

from . import related_work as rw

log = logging.getLogger(__name__)
COLS = ["R@1", "R@5", "R@10", "mAP", "mINP", "Rsum"]


def _pct(v):
    return "–" if v is None else f"{v:.2f}"


def _params_label(argv: Dict, run_file: Path) -> str:
    """How many parameters this run actually trained, for the Params column.

    This used to be ``"LoRA" if argv.get("ckpt") else "0% (zero-shot)"``, which
    is wrong in two ways that both reached ``tables_rstar/*/comparison.md``:

    * a full fine-tune also passes ``--ckpt``, so ``sdm_full_ft``,
      ``infonce_full_ft`` and ``isatmp_C_cuhk_author_sdm`` -- 149.6M
      parameters, a 598MB checkpoint -- were all printed as "LoRA";
    * IRRA is loaded through ``--irra-ckpt``, so the 100%-parameter baseline
      was printed as "0% (zero-shot)" and captioned "ours".

    The eval's own argv cannot answer the question (``--full-finetune`` is not
    passed at eval time; the loader sniffs the checkpoint). The authority is
    the TRAINING ``metrics.json`` written next to the checkpoint, which carries
    ``tuning`` and ``lora_rank``. Read that; fall back to "?" -- never to a
    guess, because a wrong Params cell is exactly the kind of error that
    survives to camera-ready.
    """
    if argv.get("irra_ckpt"):
        return "100% (IRRA)"
    if argv.get("mgcc_ckpt"):
        return "100% (MGCC)"
    ckpt = argv.get("ckpt")
    if not ckpt:
        return "0% (zero-shot)"
    tuning, rank = None, argv.get("lora_rank")
    # The ckpt path was recorded relative to the training host's cwd, so try it
    # as written and then walk up from the eval output dir (runs live at
    # <run>/occ/<condition>/metrics.json, the training one at <run>/metrics.json).
    cands = [Path(ckpt).parent] + list(run_file.parents)[:4]
    for anc in cands:
        try:
            td = json.loads((anc / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "tuning" not in td:          # that is the eval metrics.json, not train
            continue
        tuning = td.get("tuning")
        rank = td.get("lora_rank", rank)
        break
    if tuning == "full" or argv.get("full_finetune"):
        return "100% (full FT)"
    if tuning == "lora" or rank:
        return f"LoRA r{rank}" if rank else "LoRA"
    return "?"


def load_runs(paths: List[Path]) -> List[Dict]:
    """Read metrics.json files into flat rows, keeping only what a table needs."""
    out = []
    for p in paths:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("[cmp] skipping %s (%s)", p, e)
            continue
        o = d.get("overall")
        if not o:
            continue
        argv = d.get("argv") or {}
        # A run scored from a published author checkpoint is a re-scored
        # BASELINE, not our method. Labelling it venue="ours" (what this did
        # unconditionally) puts IRRA's 73.16 in the "ours" block of the table.
        is_baseline = bool(argv.get("irra_ckpt") or argv.get("mgcc_ckpt"))
        row = {
            "method": d.get("model") or argv.get("tag") or p.parent.name,
            "venue": ("re-scored baseline" if is_baseline else "ours"),
            "ours": not is_baseline,
            "backbone": f"CLIP {argv.get('pretrain', '?')}",
            "params": _params_label(argv, p),
            "benchmark": d.get("benchmark") or "?",
            "n_gallery": o.get("n_gallery"), "n_query": o.get("n_query"),
            # "official" | "reconstructed" | None (older runs, before
            # eval_protocol started copying es.meta). None is treated as
            # unknown, not as official -- see _split_caveat.
            "split_source": (d.get("meta") or {}).get("split_source"),
            "source": str(p),
        }
        # stored as fractions, published tables are in percent
        for k in COLS:
            row[k] = None if o.get(k) is None else 100 * o[k]
        out.append(row)
    return out


def build_table(runs: List[Dict], benchmark: str) -> tuple[List[Dict], str | None]:
    rows = [r for r in runs if r["benchmark"] == benchmark]
    for pub in rw.for_benchmark(benchmark):
        r = dict(pub)
        r["ours"] = False
        r["benchmark"] = benchmark
        r["Rsum"] = rw.rsum(pub)
        rows.append(r)
    # rank by R@1 so "where do we sit" is readable at a glance; unknown last
    rows.sort(key=lambda r: (r.get("R@1") is None, -(r.get("R@1") or 0)))
    return rows, rw.note_for(benchmark)


#: Marker appended to any of our rows that was NOT measured on the published
#: split. Without it the table silently invites a comparison that is not one.
_UNOFFICIAL_MARK = " \u2021"


def _split_caveat(rows: List[Dict]) -> str | None:
    """Warn when our rows and the published rows sit on different galleries.

    CUHK-PEDES is the live case: the 2017 author release ships no `split`
    field, so `prepare_cuhk.py` reconstructs val/test from the 2 000 non-train
    identities. The gallery SIZE matches the published protocol to within a
    handful of images, which is exactly what makes the resulting table
    dangerous -- the numbers look comparable and the identity membership is
    not. Say so in the table itself, not only in a log line that scrolls away.
    """
    bad = {r.get("split_source") for r in rows if r.get("ours")}
    bad = {b for b in bad if b and b != "official"}
    if not bad:
        return None
    return (f"\u2021 **Our row(s) were measured on a "
            f"{'/'.join(sorted(bad))} test split, not the authors'.** The "
            f"gallery size matches the published protocol but the identity "
            f"membership does not, so the gap to the published rows in this "
            f"table is NOT a like-for-like result and must not be quoted as "
            f"one. See "
            f"`Dataset/CUHK-PEDES/cuhk_provenance.json`; obtain the official "
            f"`reid_raw.json` and rebuild the eval set to make this table "
            f"comparable.")


def to_markdown(rows: List[Dict], benchmark: str, note: str | None) -> str:
    head = ["Method", "Venue", "Backbone", "Params", *COLS]
    body = []
    for r in rows:
        name = f"**{r['method']}**" if r.get("ours") else r["method"]
        if r.get("ours") and (r.get("split_source") or "official") != "official":
            name += _UNOFFICIAL_MARK
        body.append([name, r.get("venue", "–"), r.get("backbone", "–"),
                     r.get("params", "–"), *[_pct(r.get(c)) for c in COLS]])
    md = [f"## {benchmark} — text-to-image retrieval (%)", "",
          "| " + " | ".join(head) + " |",
          "|" + "|".join("---" for _ in head) + "|"]
    md += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]

    caveat = _split_caveat(rows)
    if caveat:
        md += ["", "> " + caveat]

    ours = [r for r in rows if r.get("ours")]
    if ours:
        g = {(r.get("n_gallery"), r.get("n_query")) for r in ours}
        md += ["", f"*Our runs: gallery/query = " +
               ", ".join(f"{a}/{b}" for a, b in sorted(g, key=str)) + ".*"]
    if note:
        md += ["", f"> **No published protocol for `{benchmark}`.** {note} "
               f"The rows above are ours only; nothing here is a SOTA claim."]
    else:
        md += ["", "> Published rows are **not** re-run by us; they are quoted. "
               "Verify each against the source PDF before submission. "
               "A blank means the paper did not report that metric."]
        srcs = sorted({r["source"] for r in rows if not r.get("ours")})
        md += ["", "Sources: " + "; ".join(srcs)]
    return "\n".join(md) + "\n"


def to_chart(rows: List[Dict], benchmark: str, out_png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log.warning("[cmp] matplotlib not installed; skipping %s", out_png.name)
        return False

    metrics = [c for c in COLS if c != "Rsum"
               and any(r.get(c) is not None for r in rows)]
    if not metrics or not rows:
        return False
    # Rsum is excluded on purpose: it is ~3x the scale of the others and would
    # flatten every bar next to it into noise.
    x = np.arange(len(metrics))
    n = len(rows)
    w = min(0.8 / max(n, 1), 0.18)

    # Published work gets a cool, muted ramp; only our run is warm. The default
    # matplotlib cycle puts a red at position 4, which lands on a published
    # method and makes it read as ours -- the one confusion this chart must not
    # create.
    ramp = plt.get_cmap("Blues")
    pub_i, n_pub = 0, sum(1 for r in rows if not r.get("ours"))

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(metrics) + 0.5 * n), 4.6))
    for i, r in enumerate(rows):
        vals = [r.get(m) if r.get(m) is not None else 0 for m in metrics]
        off = (i - (n - 1) / 2) * w
        if r.get("ours"):
            colour = "#c1272d"
        else:
            colour = ramp(0.30 + 0.55 * (pub_i / max(n_pub - 1, 1)))
            pub_i += 1
        lbl = r["method"] + (" (ours)" if r.get("ours") else "")
        if r.get("ours") and (r.get("split_source") or "official") != "official":
            lbl += _UNOFFICIAL_MARK
        bars = ax.bar(x + off, vals, w,
                      label=lbl,
                      color=colour,
                      edgecolor="black" if r.get("ours") else "none",
                      linewidth=1.2 if r.get("ours") else 0,
                      zorder=3 if r.get("ours") else 2)
        # a missing metric must not read as a zero-height bar that happens to
        # sit at the axis -- mark it
        for b, m in zip(bars, metrics):
            if r.get(m) is None:
                ax.text(b.get_x() + b.get_width() / 2, 1.0, "n/r", ha="center",
                        va="bottom", fontsize=6, rotation=90, color="#888")

    ax.set_xticks(x, metrics)
    ax.set_ylabel("%")
    title = f"{benchmark} \u2014 text-to-image retrieval"
    # A PNG is the artifact most likely to end up in a slide with no caption,
    # so the caveat has to live inside the image, not beside it.
    if any(r.get("ours") and (r.get("split_source") or "official") != "official"
           for r in rows):
        title += "\n\u2021 our row(s) on a RECONSTRUCTED split \u2014 not comparable"
        ax.title.set_fontsize(10)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return True


def write_all(rows: List[Dict], benchmark: str, note: str | None,
              out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "comparison.md"
    md.write_text(to_markdown(rows, benchmark, note), encoding="utf-8")
    csv_p = out_dir / "comparison.csv"
    with open(csv_p, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["method", "venue", "backbone", "params", "ours",
                      "split_source", *COLS, "source"])
        for r in rows:
            wtr.writerow([r["method"], r.get("venue"), r.get("backbone"),
                          r.get("params"), bool(r.get("ours")),
                          r.get("split_source") or ("official" if not
                                                    r.get("ours") else ""),
                          *[r.get(c) for c in COLS], r.get("source", "")])
    png = out_dir / "comparison.png"
    has_png = to_chart(rows, benchmark, png)
    return {"markdown": str(md), "csv": str(csv_p),
            "chart": str(png) if has_png else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics", nargs="*", default=[],
                    help="one or more metrics.json")
    ap.add_argument("--results", nargs="*", default=[],
                    help="scan one or more results trees for metrics.json "
                         "instead. Several roots are allowed so a dated run "
                         "directory (Result/<yymmdd>/) can be tabled together "
                         "with the legacy Result/occ_study/ baselines.")
    ap.add_argument("--benchmark", default=None,
                    help="restrict to one benchmark (default: one table each)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: next to the metrics file)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = [Path(p) for p in args.metrics]
    for root in args.results:
        paths += sorted(Path(root).rglob("metrics.json"))
    # One physical file can be reached through two roots (a dated dir that is
    # nested inside the legacy tree, say). Duplicated rows silently double a
    # method in the table, so de-duplicate by resolved path, keeping order.
    seen, uniq = set(), []
    for q in paths:
        k = str(q.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(q)
    paths = uniq
    if not paths:
        raise SystemExit("pass --metrics <file> or --results <dir>")

    runs = load_runs(paths)
    if not runs:
        raise SystemExit("no usable metrics.json (need an 'overall' block)")

    benches = ([args.benchmark] if args.benchmark
               else sorted({r["benchmark"] for r in runs}))
    for b in benches:
        rows, note = build_table(runs, b)
        out = Path(args.out) if args.out else Path(
            [r for r in runs if r["benchmark"] == b][0]["source"]).parent
        if args.out and len(benches) > 1:
            out = out / b
        info = write_all(rows, b, note, out)
        print(f"\n{b}: {len(rows)} row(s) -> {out}")
        for k, v in info.items():
            if v:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
