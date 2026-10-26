#!/usr/bin/env python3
"""MAINTAINER tool: collect the checkpoints behind the paper into checkpoints/
(or any --out folder ready for upload), named <config name>.pt, and check every
file against the SHA-256 recorded in configs/train/<name>.yaml.

    python tools/package_checkpoints.py --result-root ../Dataset/Result --out release_ckpts
    python tools/package_checkpoints.py --result-root ../Dataset/Result --check-only

For every config it looks in <result-root>/<provenance.original_run>/ for, in order,
provenance.original_file, lora_ckpt.pt and lora_ckpt_last.pt, and reports which one
carries the recorded hash. Run --check-only on the training server before uploading:
it proves that the files you publish are the ones the evaluation used (the eval
logs record the path <run>/lora_ckpt.pt, while some local copies were renamed).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", required=True, help="the private results tree (parent of 260915/, 261017/, ...)")
    ap.add_argument("--out", default=str(REPO / "checkpoints"))
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.result_root), Path(a.out)
    bad, sums = 0, []
    for cfg_p in sorted((REPO / "configs" / "train").glob("*.yaml")):
        cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
        ck, prov = cfg["checkpoint"], cfg["provenance"]
        run = Path(prov["original_run"])
        run = root / Path(*run.parts[1:]) if run.parts[0] == "Result" else root / run
        names = [prov["original_file"], "lora_ckpt.pt", "lora_ckpt_last.pt"]
        hits = []
        for n in dict.fromkeys(names):
            p = run / n
            if p.exists():
                h = sha256(p)
                hits.append((n, h == ck["sha256"], p))
        match = [p for n, ok, p in hits if ok]
        status = "OK " if match else "!! "
        print(f"{status}{cfg['name']:14s} " + ", ".join(f"{n}={'match' if ok else 'differs'}" for n, ok, _ in hits)
              + ("" if hits else f"no file under {run}"))
        if not match:
            if ck.get("release_file"):
                bad += 1
            continue
        if ck.get("release_file") and not a.check_only:
            out.mkdir(parents=True, exist_ok=True)
            dst = out / f"{cfg['name']}.pt"
            if not dst.exists():
                shutil.copy2(match[0], dst)
            sums.append(f"{ck['sha256']}  {cfg['name']}.pt")
    if sums:
        (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
        print(f"wrote {len(sums)} checkpoint(s) + SHA256SUMS to {out}")
    if bad:
        print(f"\n{bad} released checkpoint(s) have no file with the recorded hash. Do NOT upload until "
              "resolved: either the local copy is not the evaluated file, or the config hash is stale.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
