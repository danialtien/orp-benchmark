"""JSON / JSONL (image, caption) adapters for the caption-annotated datasets.

Three of the eligible datasets ship their captions as JSON, not parquet:

    ICFG-PEDES   list[{file_path, captions[list], split, id}]     (person crops)
    ForestPerson list[{file_path, captions[list], split, fields}] (forest, domain)
    ForestSynth  JSONL: {image, place, persons[{caption, bbox_yolo, ...}]}
                                                     (synthetic drone/forest)

All three are reduced to the same ``{image, prompt, image_id}`` pair list the
encoders consume, so they train through the identical CLIP+LoRA path as MALS /
the parquet sets.

ForestSynth is a *scene* with possibly several composited people, each carrying
its own caption + normalised YOLO bbox. To keep the (image, caption) supervision
tight, each person is cropped once (cached under ``crop_out``) and paired with
its own caption. The flat list datasets already point at single-person crops, so
they are used as-is.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)

_FILE_KEYS = ("file_path", "file_name", "image", "img_path", "path",
              "image_path")
_CAP_KEYS = ("captions", "caption", "text", "prompt")
#: per-image key, preferred over the identity field for `image_id`
_IMGID_KEYS = ("image_id", "id")
#: identity key. Emitted as ``group`` so ``train_lora`` can build
#: identity-disjoint batches.
#:
#: This matters more than it looks. In-batch InfoNCE labels every off-diagonal
#: pair a negative, and the hard-negative margin then explicitly pushes the most
#: confusable off-diagonal pair apart. On ICFG (34 674 images over 3 102
#: identities, ~11 images each) two crops of the *same person* landing in one
#: batch is common -- and the loss then trains the model to separate them. The
#: grouping already written for NOMAD fixes this for free; these datasets just
#: never told it what the identities were.
_GROUP_KEYS = ("person_id", "pid", "id")


def _get(d: dict, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _as_caption_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, (list, tuple)):
        return [str(c).strip() for c in v if c and str(c).strip()]
    return [str(v).strip()]


def _resolve(image_root: Path, rel: str) -> str:
    p = rel if os.path.isabs(rel) else str(image_root / rel)
    return p


def _crop_person(scene_path: str, bbox_yolo, crop_out: Path, tag: str):
    """Crop a normalised-YOLO (cx,cy,w,h) box out of a scene image, cached."""
    from PIL import Image
    out = crop_out / f"{tag}.jpg"
    if out.exists():
        return str(out)
    try:
        im = Image.open(scene_path).convert("RGB")
    except (FileNotFoundError, OSError):
        return None
    W, H = im.size
    cx, cy, w, h = bbox_yolo
    x1 = max(0, int((cx - w / 2) * W)); y1 = max(0, int((cy - h / 2) * H))
    x2 = min(W, int((cx + w / 2) * W)); y2 = min(H, int((cy + h / 2) * H))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop_out.mkdir(parents=True, exist_ok=True)
    im.crop((x1, y1, x2, y2)).save(out, quality=95)
    return str(out)


def load_json_pairs(
    json_path: str,
    image_root: str,
    limit: int | None = None,
    seed: int = 0,
    split_filter: tuple[str, str] | None = None,
    crop_out: str | None = None,
    require_exists: bool = True,
    oversample: int = 3,
) -> List[Dict]:
    """Return ``{image, prompt, image_id}`` pairs from a .json list or .jsonl.

    Parameters
    ----------
    json_path   : the annotation file (``*.json`` list or ``*.jsonl``).
    image_root  : directory the record image paths are relative to.
    split_filter: ``(field, value)`` to keep only e.g. ``("split", "train")``.
    crop_out    : for nested (scene, persons[]) records with bounding boxes,
                  where cropped person ROIs are cached (defaults next to images).
    """
    json_path = Path(json_path)
    image_root = Path(image_root)
    if not json_path.is_file():
        raise FileNotFoundError(f"annotation file not found: {json_path}")
    is_jsonl = json_path.suffix.lower() == ".jsonl"

    if is_jsonl:
        records = [json.loads(ln) for ln in json_path.read_text(
            encoding="utf-8").splitlines() if ln.strip()]
    else:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else (
            data.get("annotations") or data.get("images") or
            next(iter(data.values())))

    crop_dir = Path(crop_out) if crop_out else (image_root.parent / "_person_crops")
    # avoid decoding/cropping an entire scene set just to sample `limit` pairs
    scan_cap = limit * max(1, oversample) if limit else None
    pairs: List[Dict] = []
    for ri, rec in enumerate(records):
        if scan_cap is not None and len(pairs) >= scan_cap:
            break
        if split_filter is not None:
            field, val = split_filter
            if field in rec and str(rec[field]) != str(val):
                continue
        persons = rec.get("persons") if isinstance(rec, dict) else None
        if persons:                                   # nested scene -> crop each
            scene_rel = _get(rec, _FILE_KEYS) or rec.get("image")
            scene = _resolve(image_root, scene_rel)
            for pi, person in enumerate(persons):
                caps = _as_caption_list(_get(person, _CAP_KEYS))
                if not caps:
                    continue
                bbox = person.get("bbox_yolo") or person.get("bbox")
                if bbox and len(bbox) == 4:
                    img = _crop_person(scene, bbox, crop_dir,
                                       f"{Path(scene_rel).stem}_p{pi}")
                else:
                    img = scene
                if img is None or (require_exists and not os.path.exists(img)):
                    continue
                for cap in caps:
                    pairs.append({"image": img, "prompt": cap,
                                  "image_id": f"{Path(scene_rel).stem}_p{pi}"})
        else:                                         # flat single-person record
            rel = _get(rec, _FILE_KEYS)
            if rel is None:
                continue
            img = _resolve(image_root, rel)
            if require_exists and not os.path.exists(img):
                continue
            caps = _as_caption_list(_get(rec, _CAP_KEYS))
            img_id = str(_get(rec, _IMGID_KEYS) or rel)
            grp = _get(rec, _GROUP_KEYS)
            for cap in caps:
                p = {"image": img, "prompt": cap, "image_id": img_id}
                if grp is not None:
                    p["group"] = f"{json_path.stem}:{grp}"   # namespaced: two
                    # datasets both numbering identities from 0 must not merge
                pairs.append(p)

    if not pairs:
        raise RuntimeError(f"no (image, caption) pairs from {json_path} "
                           f"(image_root={image_root}); check paths/split")
    log.info("[json] %d pair(s) from %s", len(pairs), json_path.name)

    if limit is not None and limit < len(pairs):
        pairs = random.Random(seed).sample(pairs, limit)
        log.info("[json] sampled down to %d pair(s)", limit)
    return pairs


__all__ = ["load_json_pairs"]
