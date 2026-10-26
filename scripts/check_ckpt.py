#!/usr/bin/env python3
"""Inspect LoRA checkpoints before spending GPU hours evaluating them.

    python scripts/check_ckpt.py ../SAR_Pipeline/src/weight/**/lora_ckpt.pt

Prints, per checkpoint: inferred LoRA rank, tensor count, parameter count, and
the L2 norm of the B matrices. Reads the file only -- no CUDA, no model build.

Read the `|B|` column first. LoRA is initialised with **B = 0**, so the adapter
contributes exactly nothing until training moves it. ``|B| = 0`` therefore means
the checkpoint is an untrained adapter and will score exactly zero-shot CLIP, no
matter how correctly it loads. Two different checkpoints that both report
zero-shot numbers are explained either by this, or by a rank mismatch at load
time -- and this script distinguishes them in one second instead of one epoch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def describe(path: Path) -> dict:
    import torch
    from sar_person.encoders import infer_lora_rank

    ck = torch.load(str(path), map_location="cpu")
    state = ck.get("trainable", ck) if isinstance(ck, dict) else ck
    tensors = {k: v for k, v in state.items() if hasattr(v, "shape")}
    a = [v for k, v in tensors.items() if k.endswith(".A")]
    b = [v for k, v in tensors.items() if k.endswith(".B")]
    bn = sum(float(v.float().pow(2).sum()) for v in b) ** 0.5 if b else float("nan")
    an = sum(float(v.float().pow(2).sum()) for v in a) ** 0.5 if a else float("nan")
    return {
        "path": path,
        "rank": infer_lora_rank(state),
        "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
        "n_tensors": len(tensors),
        "n_lora": len(a),
        "params": sum(v.numel() for v in tensors.values()),
        "normA": an,
        "normB": bn,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpts", nargs="+")
    args = ap.parse_args()

    rows, bad = [], 0
    for p in args.ckpts:
        p = Path(p)
        try:
            rows.append(describe(p))
        except Exception as e:                                   # noqa: BLE001
            print(f"!! {p}: {e}")
            bad += 1

    if not rows:
        return 1
    w = max(len(str(r["path"])) for r in rows)
    print(f"{'checkpoint'.ljust(w)}  rank  ep   tensors  lora   params      "
          f"|A|      |B|")
    for r in sorted(rows, key=lambda r: str(r["path"])):
        flag = "  <-- B=0, UNTRAINED" if r["normB"] == 0 else ""
        print(f"{str(r['path']).ljust(w)}  {str(r['rank'] or '?'):>4}  "
              f"{str(r['epoch'] if r['epoch'] is not None else '-'):>3}  "
              f"{r['n_tensors']:>7}  {r['n_lora']:>4}  {r['params']:>9,}  "
              f"{r['normA']:>7.2f}  {r['normB']:>7.2f}{flag}")

    ranks = {r["rank"] for r in rows if r["rank"]}
    if len(ranks) > 1:
        print(f"\nMixed ranks {sorted(ranks)} in this set. eval_protocol reads "
              f"the rank off each checkpoint, so a single --lora-rank for the "
              f"whole sweep is not needed (and is ignored when it disagrees).")
    print("\nalpha is NOT stored in any checkpoint -- it only scales the "
          "forward pass. Take it from the run's metrics.json; runs produced by "
          "sar_person.run used the default alpha=16.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
