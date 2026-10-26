"""Download the remaining text-to-image person datasets for CLIP+LoRA training.

MALS and CUHK-PEDES are already present under ../Dataset. This fetches the two
that ship on the HuggingFace Hub as parquet-with-embedded-images (so no gated
image-request form is needed), and prints exact instructions for the two that do
NOT have a public auto-downloadable mirror.

Run from the SAR_Person project root:

    python scripts/download_datasets.py                 # ICFG + RSTP
    python scripts/download_datasets.py --only rstp
    python scripts/download_datasets.py --dest ../Dataset

Needs `huggingface_hub` (`pip install huggingface_hub`). You are already logged
in as your HF user, or run `huggingface-cli login` first.

After downloading, train with e.g.:
    python -m sar_person.train_dataset --dataset rstp --encoder clip --out ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# HuggingFace repos that carry parquet with embedded images (auto-downloadable).
HF_DATASETS = {
    "icfg": {
        "repo_id": "cjc/ICFG-PEDES",
        "dest": "ICFG-PEDES",
        "note": "If the viewer/parquet export is empty, try the alternate mirror "
                "'TruongVox/ICFG-PEDES-dataset'.",
    },
    "rstp": {
        "repo_id": "tuandunghcmut/RSTPReid",
        "dest": "RSTPReid",
        "note": "Columns: image, captions[list], id, file_path, split. "
                "train_dataset filters split=='train' automatically.",
    },
}

# Datasets with no public one-command mirror — manual steps, with sources.
MANUAL = {
    "ufinebench": [
        "UFineBench / UFine6926 (ultra-fine captions, 26,206 imgs / 52,412 texts):",
        "  GitHub:  https://github.com/Zplusdragon/UFineBench",
        "  Follow its README to request/download UFine6926, then place the images",
        "  + annotation json under ../Dataset/UFineBench/ and adapt a loader (the",
        "  parquet loader works if you export it to parquet with image+caption).",
    ],
    "pdestre": [
        "P-DESTRE (aerial/UAV pedestrian, attributes + multi-day IDs):",
        "  Project/DL: http://p-destre.di.ubi.pt/  (University of Beira Interior)",
        "  arXiv:      https://arxiv.org/abs/2004.02782",
        "  It ships as UAV VIDEOS + per-frame annotations (no free-text captions).",
        "  To use for CLIP+LoRA: (1) extract frames, (2) crop persons from the box",
        "  annotations, (3) build prompts from the soft-biometric/clothing labels",
        "  via SAR_Person's PromptStyle templates, then feed as (image, prompt).",
    ],
}


def download_hf(repo_id: str, dest: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ! huggingface_hub not installed. Run: pip install huggingface_hub",
              file=sys.stderr)
        return False
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {repo_id} -> {dest} ...")
    snapshot_download(repo_id=repo_id, repo_type="dataset",
                      local_dir=str(dest), local_dir_use_symlinks=False)
    print(f"  done: {dest}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[2]
                                          / "Dataset"),
                    help="Dataset root (default: ../Dataset)")
    ap.add_argument("--only", nargs="+", choices=list(HF_DATASETS),
                    help="download only these HF datasets (default: all)")
    args = ap.parse_args()

    root = Path(args.dest)
    targets = args.only or list(HF_DATASETS)
    print(f"== HuggingFace parquet datasets -> {root} ==")
    ok, fail = [], []
    for key in targets:
        spec = HF_DATASETS[key]
        print(f"[{key}] {spec['repo_id']}")
        print(f"       {spec['note']}")
        try:
            (ok if download_hf(spec["repo_id"], root / spec["dest"]) else fail
             ).append(key)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed: {exc}", file=sys.stderr)
            print("  (gated or network-blocked — open the repo page and accept "
                  "terms, or download manually)", file=sys.stderr)
            fail.append(key)

    print("\n== Manual-download datasets (no public one-command mirror) ==")
    for lines in MANUAL.values():
        for ln in lines:
            print(ln)
        print()

    print(f"summary: downloaded={ok or '-'}  failed={fail or '-'}")
    print("next: python -m sar_person.train_dataset --dataset <icfg|rstp> "
          "--encoder clip --out ../Dataset/Test_SAR_Person/<name>_lora")


if __name__ == "__main__":
    main()
