#!/usr/bin/env python3
"""MAINTAINER tool. Export, per benchmark, the identity structure needed to
re-run every statistic without the dataset: for each query (in evaluation
order) an anonymised identity index and k = |G_q| (number of gallery images of
that identity), plus the gallery size N. No text, no file names, no pixels.

Identity indices are ranks of the sorted original pid strings, so they carry no
information beyond the grouping."""
import argparse, collections, json
from pathlib import Path
import numpy as np

SRC = {"cuhk": "occluded/manifests/cuhk_foliage_l10.json",
       "rstp": "occluded/manifests/rstp_foliage_l10.json",
       "icfg": "occluded/manifests/icfg_foliage_l10.json"}

ap = argparse.ArgumentParser()
ap.add_argument("--result-root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--nomad-metrics", default="261002/drone/eval/Docc/nomad_gallery_val/metrics.json")
a = ap.parse_args()
root = Path(a.result_root)
out = {}
for b, rel in SRC.items():
    m = json.loads((root / rel).read_text(encoding="utf-8"))
    qp = [q["pid"] for q in m["queries"]]
    gc = collections.Counter(g["pid"] for g in m["gallery"])
    ids = sorted(set(qp))
    idx = {p: i for i, p in enumerate(ids)}
    out[f"{b}_qid"] = np.array([idx[p] for p in qp], np.int32)
    out[f"{b}_k"] = np.array([gc[p] for p in qp], np.int32)
    out[f"{b}_N"] = np.int32(len(m["gallery"]))
    print(b, "queries", len(qp), "ids", len(ids), "gallery", len(m["gallery"]), "mean k %.3f" % out[f"{b}_k"].mean())
# NOMAD: one query per held-out actor -> identity = query
d = json.loads((root / a.nomad_metrics).read_text(encoding="utf-8"))
n = len(d["per_query"]["inp"])
out["nomad_qid"] = np.arange(n, dtype=np.int32)
out["nomad_N"] = np.int32(d["overall"]["n_gallery"])
print("nomad queries", n, "gallery", int(out["nomad_N"]))
np.savez_compressed(a.out, **out)
