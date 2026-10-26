#!/usr/bin/env python3
"""Check that your prepared TRAINING data is the data the paper trained on.

    python scripts/verify_training_data.py --data-root ../Dataset            # both
    python scripts/verify_training_data.py --data-root ../Dataset --only cuhk

Compares
  * Dataset/CUHK-PEDES/annotations/train.jsonl  (preset `cuhk_author`, built by
    `python scripts/prepare_cuhk.py --rebuild-train` from the official reid_raw.json)
    against manifests/train/cuhk_author_train.json.gz: same images in the same
    order, same identity labels, same caption text (by SHA-1);
  * Dataset/NOMAD/crop/annotations.json  (preset `nomad`, built by
    `python scripts/prepare_nomad.py`) against manifests/train/nomad_crop.json.gz:
    same crops, same actor split, same source boxes, same actor descriptions.

The manifests hold no caption text and no pixels. Standard library only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def load_gz(p: Path):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def check_cuhk(root: Path) -> bool:
    ref = load_gz(REPO / "manifests" / "train" / "cuhk_author_train.json.gz")
    p = root / "CUHK-PEDES" / "annotations" / "train.jsonl"
    if not p.exists():
        print(f"[cuhk] {p} not found -- run: python scripts/prepare_cuhk.py --data-root {root} --rebuild-train")
        return False
    mine = []
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        mine.append([r["image_id"], r["person_id"], [sha1(c) for c in r["captions"]]])
    ok_order = [r[0] for r in mine] == [r[0] for r in ref["rows"]]
    ok_pid = [str(r[1]) for r in mine] == [str(r[1]) for r in ref["rows"]]
    ok_cap = [r[2] for r in mine] == [r[2] for r in ref["rows"]]
    missing = sum(not (root / "CUHK-PEDES" / "imgs" / r[0]).exists() for r in mine)
    print(f"[cuhk] {len(mine)} images (paper: {ref['stats']['images']}), image order {'OK' if ok_order else 'DIFFERS'}, "
          f"identities {'OK' if ok_pid else 'DIFFER'}, captions {'OK' if ok_cap else 'DIFFER'}, "
          f"missing image files: {missing}")
    return ok_order and ok_pid and ok_cap and missing == 0


def check_nomad(root: Path) -> bool:
    ref = load_gz(REPO / "manifests" / "train" / "nomad_crop.json.gz")
    p = root / "NOMAD" / "crop" / "annotations.json"
    if not p.exists():
        print(f"[nomad] {p} not found -- run: python scripts/prepare_nomad.py --workers 8")
        return False
    a = json.loads(p.read_text(encoding="utf-8"))
    cols = ref["columns"]
    mine = {r["id"]: [r.get(c) for c in cols] for r in a}
    want = {r[0]: r for r in ref["rows"]}
    same_set = set(mine) == set(want)
    diff = [k for k in want if k in mine and mine[k] != want[k]]
    caps = {}
    for r in a:
        caps.setdefault(r["person_id"], sha1(r["prompt_default"]))
    cap_ok = caps == ref["actor_caption_sha1"]
    missing = sum(not (root / "NOMAD" / "crop" / "images" / r["file_path"]).exists() for r in a)
    print(f"[nomad] {len(mine)} crops (paper: {ref['stats']['crops']}), crop set {'OK' if same_set else 'DIFFERS'}, "
          f"rows differing in split/box/metadata: {len(diff)}, actor descriptions {'OK' if cap_ok else 'DIFFER'}, "
          f"missing crop files: {missing}")
    if diff:
        print("        e.g.", diff[:3])
    return same_set and not diff and cap_ok and missing == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--only", choices=["cuhk", "nomad"], default=None)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser().resolve()
    ok = True
    if a.only in (None, "cuhk"):
        ok &= check_cuhk(root)
    if a.only in (None, "nomad"):
        ok &= check_nomad(root)
    print("TRAINING DATA IDENTICAL TO THE PAPER" if ok else "DIFFERENCES FOUND -- see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
