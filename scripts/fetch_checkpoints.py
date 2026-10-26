#!/usr/bin/env python3
"""Download the released checkpoints into checkpoints/ and verify their SHA-256.

    python scripts/fetch_checkpoints.py                 # all 10
    python scripts/fetch_checkpoints.py D DA_s0         # a subset
    python scripts/fetch_checkpoints.py --verify-only   # hash what is already there

The download location is set once in checkpoints/README.md / BASE_URL below
(a Zenodo record or a Hugging Face model repo). Files are ~600 MB each (fp32
CLIP ViT-B/16 backbone, full fine-tune; training-only heads are not included).
Needs only the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "checkpoints"
#: Filled in when the checkpoints are published, e.g.
#: "https://zenodo.org/records/<id>/files/{name}.pt?download=1"  or
#: "https://huggingface.co/<user>/orp-checkpoints/resolve/main/{name}.pt"
BASE_URL = "TO_BE_FILLED_AT_PUBLICATION/{name}.pt"


def recorded() -> dict[str, str]:
    """name -> sha256, read from configs/train/*.yaml without needing PyYAML."""
    out = {}
    for p in sorted((REPO / "configs" / "train").glob("*.yaml")):
        t = p.read_text(encoding="utf-8")
        if re.search(r"^\s*release_file:\s*null", t, re.M):
            continue
        m = re.search(r"^\s*sha256:\s*([0-9a-f]{64})", t, re.M)
        if m:
            out[p.stem] = m.group(1)
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--base-url", default=BASE_URL)
    a = ap.parse_args()
    want = recorded()
    names = a.names or list(want)
    CKPT.mkdir(exist_ok=True)
    bad = 0
    for n in names:
        if n not in want:
            print(f"!! unknown checkpoint '{n}' (known: {', '.join(want)})")
            bad += 1
            continue
        p = CKPT / f"{n}.pt"
        if not p.exists() and not a.verify_only:
            if a.base_url.startswith("TO_BE_FILLED"):
                raise SystemExit("BASE_URL is not set yet; see checkpoints/README.md")
            url = a.base_url.format(name=n)
            print(f"[get] {url}")
            urllib.request.urlretrieve(url, p)
        if not p.exists():
            print(f"-- {n}: missing")
            bad += 1
            continue
        ok = sha256(p) == want[n]
        print(f"{'OK' if ok else '!!'} {n}: sha256 {'matches' if ok else 'DIFFERS from'} configs/train/{n}.yaml")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
