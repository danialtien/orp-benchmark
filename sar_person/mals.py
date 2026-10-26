"""MALS gene_crop adapter for the SAR_Person retrieval pipeline.

MALS (APTM) ships its generated crops as a flat image folder plus an attribute
JSON (``gene_attrs/g_4x_attrs.json``) with one record per image:

    {"image": "gene_crop/4x/0.jpg",
     "caption": "a woman standing on a set of stairs ...",
     "image_id": "4x_0",
     "label": [0, 1, -1, 1, ...]}          # 27-dim APTM attribute vector

This module turns that JSON into the ``(image, prompt)`` pair list that the rest
of the pipeline (encoders -> metrics_viz) already understands, so MALS can be
evaluated with exactly the same CLIP ViT-B/16 code path as the SAR profiles.

Each image/caption pair is treated as its own identity (standard CUHK-PEDES /
MALS text-to-image retrieval protocol): for image row i the ground-truth prompt
column is i, i.e. ``gt_col = list(range(n))`` when the pair order is preserved.
"""
from __future__ import annotations
import json, os, random
from pathlib import Path
from typing import Dict, List


def load_mals_pairs(
    attrs_json: str,
    mals_root: str | None = None,
    limit: int | None = None,
    seed: int = 0,
    stride: int = 1,
    require_exists: bool = True,
) -> List[Dict]:
    """Return a list of ``{image, prompt, image_id, label}`` dicts.

    Parameters
    ----------
    attrs_json : path to ``g_4x_attrs.json`` (or any gene_attrs JSON).
    mals_root  : dataset root the ``image`` paths are relative to. Defaults to
                 two levels up from the attrs file (``.../MALS``), which is where
                 the ``gene_crop/`` folder lives.
    limit      : keep at most this many pairs (after optional shuffling).
    seed       : RNG seed used only when ``limit`` triggers random sampling.
    stride     : take every ``stride``-th record before sampling (cheap uniform
                 thinning of the 128k set without loading images).
    require_exists : drop records whose image file is missing.
    """
    attrs_json = str(attrs_json)
    if mals_root is None:
        # gene_attrs/g_4x_attrs.json  ->  MALS/
        mals_root = str(Path(attrs_json).resolve().parents[1])
    with open(attrs_json, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    if stride > 1:
        records = records[::stride]

    pairs: List[Dict] = []
    for rec in records:
        rel = rec["image"]
        img_path = rel if os.path.isabs(rel) else os.path.join(mals_root, rel)
        if require_exists and not os.path.exists(img_path):
            continue
        pairs.append(
            {
                "image": img_path,
                "prompt": rec["caption"],
                "image_id": rec.get("image_id", rel),
                "label": rec.get("label"),
            }
        )

    if limit is not None and limit < len(pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, limit)

    if not pairs:
        raise RuntimeError(f"No usable (image, caption) pairs found in {attrs_json}")
    return pairs
