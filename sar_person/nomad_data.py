"""NOMAD crop set -> ``{image, prompt, image_id, group, split}`` training pairs.

Reads the ``annotations.json`` written by ``scripts/prepare_nomad.py`` (person
crops + one caption per actor) and reduces it to the pair list the encoders
consume. NOMAD needs its own thin adapter rather than the generic
``json_data.load_json_pairs`` for two reasons, both consequences of "many crops,
one description":

``group``  every crop of an actor carries the identical caption, so batches must
           be identity-disjoint -- otherwise in-batch InfoNCE labels two crops of
           the same person mutual negatives and the hard-negative margin pushes a
           true positive apart. ``ClipEncoder.train_lora`` honours this key.
``split``  the train/val split is over ACTORS (assigned at prepare time), so it
           has to survive into the pair list; a random crop-level holdout would
           put the same caption on both sides of the split.

Prompt flavour is selectable at load time (``prompt_style``) because
``prepare_nomad.py`` stores all three flavours in every record -- switching
DEFAULT <-> APTM templated costs nothing and needs no re-cropping.
"""
from __future__ import annotations

import json
import logging
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)

PROMPT_KEYS = {
    "default": "prompt_default",
    "templated": "prompt_templated",
    "templated_neg": "prompt_templated_neg",
}


def load_nomad_pairs(
    crop_root: str,
    prompt_style: str = "default",
    limit: int | None = None,
    seed: int = 0,
    altitudes: "list[int] | None" = None,
    min_visibility: int = 0,
    activities: "list[str] | None" = None,
    require_exists: bool = True,
) -> List[Dict]:
    """Load the prepared NOMAD crop set.

    Parameters
    ----------
    crop_root      : the ``NOMAD/crop`` directory (holds ``annotations.json``
                     and ``images/``).
    prompt_style   : ``default`` | ``templated`` | ``templated_neg``.
    limit          : max pairs to keep. Sampling is STRATIFIED BY ACTOR so a
                     small limit still sees all 100 identities -- a plain random
                     sample of 500 out of 9k crops would leave some actors with
                     one crop and others with twenty.
    altitudes      : keep only these flight altitudes in metres, e.g. ``[10]``.
    min_visibility : tighten the visibility floor beyond the prepare-time one.
    activities     : keep only these NOMAD activity labels, e.g.
                     ``["Walking", "Laying"]``.
    """
    root = Path(crop_root)
    ann_path = root / "annotations.json"
    if not ann_path.is_file():
        raise FileNotFoundError(
            f"{ann_path} not found -- run scripts/prepare_nomad.py first")
    key = PROMPT_KEYS.get(prompt_style)
    if key is None:
        raise ValueError(f"prompt_style must be one of {sorted(PROMPT_KEYS)}")

    records = json.loads(ann_path.read_text(encoding="utf-8"))
    img_root = root / "images"
    alt_set = set(altitudes) if altitudes else None
    act_set = set(activities) if activities else None

    by_actor: Dict[str, List[Dict]] = {}
    dropped = Counter()
    for rec in records:
        if alt_set is not None and rec.get("altitude_m") not in alt_set:
            dropped["altitude"] += 1
            continue
        if int(rec.get("visibility", 100)) < min_visibility:
            dropped["visibility"] += 1
            continue
        if act_set is not None and rec.get("activity") not in act_set:
            dropped["activity"] += 1
            continue
        prompt = rec.get(key) or (rec.get("captions") or [""])[0]
        if not prompt:
            dropped["no_prompt"] += 1
            continue
        img = str(img_root / rec["file_path"])
        if require_exists and not os.path.exists(img):
            dropped["missing_image"] += 1
            continue
        actor = rec["person_id"]
        by_actor.setdefault(actor, []).append({
            "image": img,
            "prompt": prompt,
            "image_id": rec["id"],
            "group": actor,          # identity-disjoint batching (see module doc)
            "eval_group": actor,     # one crop per actor in the retrieval eval
            "split": rec.get("split", "train"),
            "altitude_m": rec.get("altitude_m"),
            "visibility": rec.get("visibility"),
            "activity": rec.get("activity", ""),
        })

    if not by_actor:
        raise RuntimeError(f"no NOMAD pairs survived the filters in {ann_path} "
                           f"(dropped: {dict(dropped)})")

    pairs = _stratified(by_actor, limit, seed)
    n_val = sum(1 for p in pairs if p["split"] == "val")
    log.info("[nomad] %d pair(s) | %d actor(s) | %d train / %d val | style=%s%s",
             len(pairs), len(by_actor), len(pairs) - n_val, n_val, prompt_style,
             f" | dropped {dict(dropped)}" if dropped else "")
    return pairs


def _stratified(by_actor: Dict[str, List[Dict]], limit: int | None,
                seed: int) -> List[Dict]:
    """Even round-robin draw across actors, capped at ``limit`` in total."""
    rng = random.Random(seed)
    for lst in by_actor.values():
        rng.shuffle(lst)
    if limit is None:
        out = [p for actor in sorted(by_actor) for p in by_actor[actor]]
        rng.shuffle(out)
        return out

    out: List[Dict] = []
    actors = sorted(by_actor)
    cursor = {a: 0 for a in actors}
    while len(out) < limit:
        progressed = False
        for a in actors:
            if len(out) >= limit:
                break
            i = cursor[a]
            if i < len(by_actor[a]):
                out.append(by_actor[a][i])
                cursor[a] = i + 1
                progressed = True
        if not progressed:
            break
    rng.shuffle(out)
    return out


__all__ = ["load_nomad_pairs", "PROMPT_KEYS"]
