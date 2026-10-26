#!/usr/bin/env python3
"""Check that your local copy of each benchmark yields EXACTLY the evaluation
set used in the paper (same gallery images in the same order, same identity
labels, same queries in the same order, same caption text).

    python scripts/verify_evalsets.py --data-root ../Dataset --benchmarks cuhk icfg rstp
    python scripts/verify_evalsets.py --data-root ../Dataset --benchmarks nomad

The released references (``manifests/evalsets/<bench>.json.gz``) contain only
dataset-relative image ids, identity labels and a SHA-1 of every caption; no
caption text and no pixels. Your local evaluation set is built with the same
code the paper used (``sar_person.eval_sets.build``); pass ``--cache DIR`` to
also write it as a JSON manifest for ``sar_person.eval_protocol --evalset-cache``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sar_person.eval_sets import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=str(REPO.parent / "Dataset"))
    ap.add_argument("--benchmarks", nargs="+", default=["cuhk", "icfg", "rstp"])
    ap.add_argument("--cache", default=None, help="write the built evaluation sets here")
    a = ap.parse_args()
    bad = 0
    for b in a.benchmarks:
        ref_p = REPO / "manifests" / "evalsets" / f"{b}.json.gz"
        with gzip.open(ref_p, "rt", encoding="utf-8") as f:
            ref = json.load(f)
        es = build(b, a.data_root, cache_dir=a.cache)
        g_loc = [[g["image_id"], str(g["pid"])] for g in es.gallery]
        q_loc = [[q["image_id"], str(q["pid"]), hashlib.sha1(q["text"].encode("utf-8")).hexdigest()]
                 for q in es.queries]
        problems = []
        if len(g_loc) != len(ref["gallery"]):
            problems.append(f"gallery size {len(g_loc)} != {len(ref['gallery'])}")
        if len(q_loc) != len(ref["queries"]):
            problems.append(f"query count {len(q_loc)} != {len(ref['queries'])}")
        n_g = sum(x == y for x, y in zip(g_loc, ref["gallery"]))
        n_q = sum(x[:2] == y[:2] for x, y in zip(q_loc, ref["queries"]))
        n_t = sum(x[2] == y[2] for x, y in zip(q_loc, ref["queries"]))
        if n_g != len(ref["gallery"]):
            problems.append(f"{len(ref['gallery']) - n_g} gallery entries differ (image id / pid / order)")
        if n_q != len(ref["queries"]):
            problems.append(f"{len(ref['queries']) - n_q} queries differ (image id / pid / order)")
        if n_t != len(ref["queries"]):
            problems.append(f"{len(ref['queries']) - n_t} caption texts differ "
                            "(different annotation file or text normalisation)")
        print(f"[{b}] gallery {len(g_loc)} images / {len({g[1] for g in g_loc})} ids, "
              f"{len(q_loc)} queries -> " + ("IDENTICAL to the paper" if not problems else "; ".join(problems)))
        bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
