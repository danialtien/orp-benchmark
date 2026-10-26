"""Parquet (image, caption) adapter for HuggingFace-style person datasets.

Several text-to-image person datasets ship on the HuggingFace Hub as parquet
shards with the image bytes embedded inline, which sidesteps the gated
image-request forms of the originals. This module turns those shards into the
same ``{image, prompt, image_id, label}`` pair list that ``mals.load_mals_pairs``
produces, so ``train_dataset`` / ``eval_mals`` can consume them with the exact
same CLIP ViT-B/16 code path.

Handled layouts (auto-detected):

    CUHK-PEDES  cols: image{bytes,path}, text                 (1 caption / row)
    MALS        cols: image{bytes,path}, caption, image_id, label
    RSTPReid    cols: image{bytes,path}, captions[list], id, file_path, split
    ICFG-PEDES  cols: image{bytes,path}, caption/text

Because the image bytes live inside the parquet, the loader materialises each
image to ``image_out_dir`` once (idempotent: existing files are not rewritten),
then returns pair dicts whose ``image`` field is an absolute path on disk — which
is what both encoders (``ClipEncoder._prep_image`` / ``MockEncoder``) expect.

Standard text-to-image retrieval protocol: each (image, caption) is its own row;
when a row carries several captions they are exploded into several pairs that
share the same image path.
"""
from __future__ import annotations

import glob
import logging
import os
import random
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)

#: caption column candidates, most specific first
_CAPTION_COLS = ("caption", "text", "captions", "prompt", "sentence")
#: image column candidates
_IMAGE_COLS = ("image", "img", "jpg", "png")


def _pick(colnames: list[str], candidates) -> str | None:
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _image_bytes_and_name(cell, row_idx: int, shard: str):
    """Return (bytes, filename) from an embedded-image cell.

    HuggingFace `Image` features and raw structs both decode via pyarrow to a
    dict ``{"bytes": b"...", "path": "0.png"}``. A few exporters store the bytes
    directly; both are handled.
    """
    if isinstance(cell, dict):
        data = cell.get("bytes")
        name = cell.get("path") or f"{shard}_{row_idx}.jpg"
    elif isinstance(cell, (bytes, bytearray)):
        data = bytes(cell)
        name = f"{shard}_{row_idx}.jpg"
    else:
        return None, None
    if not data:
        return None, None
    # flatten any sub-directories in the stored path into a unique flat name
    name = Path(str(name).replace("\\", "/")).name
    return data, name


def _captions(cell) -> list[str]:
    """Normalise a caption cell into a non-empty list of strings."""
    if cell is None:
        return []
    if isinstance(cell, str):
        return [cell] if cell.strip() else []
    if isinstance(cell, (list, tuple)):
        return [str(c).strip() for c in cell if c and str(c).strip()]
    # numpy array / arrow list -> iterate
    try:
        return [str(c).strip() for c in list(cell) if c and str(c).strip()]
    except TypeError:
        s = str(cell).strip()
        return [s] if s else []


def load_parquet_pairs(
    parquet_dir: str,
    image_out_dir: str,
    limit: int | None = None,
    seed: int = 0,
    caption_col: str | None = None,
    image_col: str | None = None,
    split_col_filter: tuple[str, str] | None = None,
    max_captions_per_image: int | None = None,
    oversample: int = 3,
    exclude_names: "set[str] | None" = None,
) -> List[Dict]:
    """Materialise embedded images and return ``{image, prompt, image_id, label}``.

    Parameters
    ----------
    parquet_dir : folder holding one or more ``*.parquet`` shards (searched
                  recursively).
    image_out_dir : where decoded images are written (created if missing).
    limit       : keep at most this many pairs (random sample, after exploding
                  captions).
    seed        : RNG seed for the ``limit`` sample.
    caption_col / image_col : override auto-detection.
    split_col_filter : ``(column, value)`` to keep only rows where
                  ``row[column] == value`` (e.g. ``("split", "train")``).
    max_captions_per_image : cap captions kept per image (None = all).
    exclude_names : image filenames (as stored in the parquet ``path`` field) to
                  drop. Used to keep a self-made evaluation holdout out of the
                  training pairs -- see ``eval_sets._write_exclusion_list``.
    """
    import pyarrow.parquet as pq
    exclude_names = set(exclude_names or ())

    shards = sorted(glob.glob(os.path.join(parquet_dir, "**", "*.parquet"),
                             recursive=True))
    if not shards:
        raise FileNotFoundError(f"no *.parquet under {parquet_dir}")
    out_dir = Path(image_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # to avoid materialising an entire 10^5-row dataset just to sample `limit`
    # pairs, stop once we have a healthy oversample of the target (then sample).
    scan_cap = limit * max(1, oversample) if limit else None

    pairs: List[Dict] = []
    written = 0
    excluded = 0
    for shard_path in shards:
        if scan_cap is not None and len(pairs) >= scan_cap:
            break
        stem = Path(shard_path).stem
        table = pq.read_table(shard_path)
        cols = table.column_names
        icol = image_col or _pick(cols, _IMAGE_COLS)
        ccol = caption_col or _pick(cols, _CAPTION_COLS)
        if icol is None or ccol is None:
            raise ValueError(
                f"{shard_path}: could not find image/caption columns in {cols}. "
                f"Pass --image-col / --caption-col explicitly.")
        has_id = _pick(cols, ("image_id", "id", "file_path", "path"))
        # Identity label, separate from image_id. json_data emits "group" from
        # person_id/pid/id; without the same key here every parquet pair looks
        # like ONE identity to encoders.train_lora -- the SDM label matrix, the
        # PK sampler, the ID classifier and the --eval-ids identity holdout all
        # read it, and nothing warns when it is missing.
        has_pid = _pick(cols, ("person_id", "pid", "id", "label"))
        has_label = "label" in cols
        # only apply the split filter when the shard actually carries that column
        # (HF parquet exports sometimes split by folder instead of a column)
        apply_split = split_col_filter is not None and split_col_filter[0] in cols
        if split_col_filter is not None and not apply_split:
            # Silently returning every row when the caller asked for "test" is
            # how a train/test leak gets into a results table unnoticed.
            log.warning(
                "[parquet] %s: asked for %s=%r but there is no %r column "
                "(cols: %s) -- returning ALL rows UNFILTERED. If this shard is "
                "train-only, whatever consumes it is NOT evaluating on a "
                "held-out split.", Path(shard_path).name, split_col_filter[0],
                split_col_filter[1], split_col_filter[0], cols)
        rows = table.to_pylist()
        for i, row in enumerate(rows):
            if scan_cap is not None and len(pairs) >= scan_cap:
                break
            if apply_split:
                col, val = split_col_filter
                if str(row.get(col)) != str(val):
                    continue
            data, name = _image_bytes_and_name(row.get(icol), i, stem)
            if data is None:
                continue
            if name in exclude_names:
                excluded += 1
                continue
            # keep shard stem in the name so identically-named files across
            # shards do not collide
            fname = f"{stem}__{name}"
            fpath = out_dir / fname
            if not fpath.exists():
                fpath.write_bytes(data)
                written += 1
            caps = _captions(row.get(ccol))
            if max_captions_per_image is not None:
                caps = caps[:max_captions_per_image]
            img_id = str(row.get(has_id)) if has_id else fname
            for cap in caps:
                p = {
                    "image": str(fpath),
                    "prompt": cap,
                    "image_id": img_id,
                    "label": row.get("label") if has_label else None,
                }
                if has_pid is not None and row.get(has_pid) is not None:
                    # namespaced by shard stem: two datasets both numbering
                    # identities from 0 must not merge
                    p["group"] = f"{stem}:{row.get(has_pid)}"
                pairs.append(p)

    if not pairs:
        raise RuntimeError(f"no (image, caption) pairs materialised from "
                           f"{parquet_dir}")
    log.info("[parquet] %d pair(s) from %d shard(s); %d new image(s) -> %s",
             len(pairs), len(shards), written, out_dir)
    if exclude_names:
        log.info("[parquet] dropped %d row(s) belonging to %d held-out image(s)",
                 excluded, len(exclude_names))
        if not excluded:
            log.warning("[parquet] an exclusion list was given but NOTHING "
                        "matched -- the names in it do not look like this "
                        "dataset's image paths, so the holdout is NOT protected")

    if limit is not None and limit < len(pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, limit)
        log.info("[parquet] sampled down to %d pair(s) (seed=%d)", limit, seed)
    return pairs


__all__ = ["load_parquet_pairs"]
