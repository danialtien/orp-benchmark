#!/usr/bin/env python3
"""Build the RSTPReid test evaluation set from the OFFICIAL release.

The paper's code path (sar_person.eval_sets._load_rstp) reads RSTPReid from a
parquet export with the images embedded. If you obtained RSTPReid from its
authors instead (https://github.com/NjtechCVLab/RSTPReid-Dataset), you have

    Dataset/RSTPReid/imgs/<file>.jpg
    Dataset/RSTPReid/data_captions.json      [{id, img_path, captions, split}, ...]

This script turns that into the cached evaluation set that eval_protocol,
occlude_bench and verify_occlusion read, in EXACTLY the paper's gallery and
query order (taken from manifests/evalsets/rstp.json.gz). Every query caption is
matched to the released SHA-1, so a successful run proves the evaluation set is
identical to the paper's.

    python scripts/prepare_rstp_official.py --data-root ../Dataset --cache runs/evalsets
    # then pass  --evalset-cache runs/evalsets  to eval_protocol / occlude_bench
    # (tools/run_config.py does this for you with --out-root runs)

Needs only the Python standard library.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cache", required=True, help="evalset cache dir (written: rstp.json)")
    ap.add_argument("--annotation", default=None, help="default <data-root>/RSTPReid/data_captions.json")
    ap.add_argument("--images", default=None, help="default <data-root>/RSTPReid/imgs")
    a = ap.parse_args()
    root = Path(a.data_root).expanduser().resolve()
    ann = Path(a.annotation) if a.annotation else root / "RSTPReid" / "data_captions.json"
    imgs = Path(a.images) if a.images else root / "RSTPReid" / "imgs"
    recs = json.loads(ann.read_text(encoding="utf-8"))
    by_name = {}
    for r in recs:
        if str(r.get("split")) != "test":
            continue
        name = Path(str(r.get("img_path") or r.get("file_path"))).name
        by_name[name] = r
    with gzip.open(REPO / "manifests" / "evalsets" / "rstp.json.gz", "rt", encoding="utf-8") as f:
        ref = json.load(f)

    gallery, missing = [], []
    for iid, pid in ref["gallery"]:
        r = by_name.get(Path(iid).name)
        p = imgs / Path(str(r.get("img_path"))) if r else imgs / iid
        if not p.exists():
            missing.append(str(p))
        if r and str(r.get("id")) != str(pid):
            raise SystemExit(f"identity mismatch for {iid}: official id {r.get('id')} vs released {pid}")
        gallery.append({"image": str(p), "pid": str(pid), "image_id": iid})
    if missing:
        raise SystemExit(f"{len(missing)} gallery image(s) not found, e.g. {missing[:3]}")

    caps = defaultdict(dict)                         # image_id -> {sha1: text}
    for name, r in by_name.items():
        for c in r.get("captions") or []:
            t = str(c).strip()
            if t:
                caps[name][sha1(t)] = t
    queries, bad = [], 0
    for iid, pid, h in ref["queries"]:
        t = caps[Path(iid).name].get(h)
        if t is None:
            bad += 1
            continue
        queries.append({"text": t, "pid": str(pid), "image_id": iid})
    if bad:
        raise SystemExit(f"{bad} of {len(ref['queries'])} query caption(s) do not match the released SHA-1. "
                         "Your annotation differs from the one used in the paper.")

    out = Path(a.cache) / "rstp.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    gp, qp = {g["pid"] for g in gallery}, {q["pid"] for q in queries}
    out.write_text(json.dumps({
        "version": 1, "name": "rstp",
        "meta": {"dataset": "RSTPReid", "split": "test", "source": str(ann),
                 "built_by": "scripts/prepare_rstp_official.py (released order)",
                 "expected": {"gallery_images": 1000, "gallery_ids": 200, "queries": 2000}},
        "stats": {"gallery_images": len(gallery), "gallery_ids": len(gp), "queries": len(queries),
                  "query_ids": len(qp), "queries_without_gallery_match": len(qp - gp),
                  "avg_gallery_per_query_id": round(len(gallery) / max(len(qp), 1), 2)},
        "gallery": gallery, "queries": queries}, indent=1), encoding="utf-8")
    print(f"[rstp] {len(gallery)} gallery images / {len(gp)} ids, {len(queries)} queries "
          f"-> IDENTICAL to the paper (all captions match) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
