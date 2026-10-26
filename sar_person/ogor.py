"""OGor -- MGCC's Occlusion Generator, reproduced from the released code.

MGCC (Wu et al., AAAI-24) builds Occluded-CUHK-PEDES / Occluded-ICFG-PEDES /
Occluded-RSTPReid by pasting cut-out object instances onto person crops. Their
numbers are only comparable to ours if our generator is *their* generator, so
this module reproduces ``dataset/process_data.py::Occlusion_Adding`` from
https://github.com/littlexinyi/MGCC rather than the prose in the paper.

Why that distinction matters -- the code and the paper disagree, twice
--------------------------------------------------------------------
1. **How many images get occluded.** The paper says "we randomly select 30% of
   the whole train, val, and test images within the same ID but different
   views". The code says::

       if last_id != current_id:   Occlusion(...)      # occlude
       else:                       holistic_img.save() # copy verbatim

   i.e. exactly *one image per identity*, the first in file order, is occluded.
   With 5-10 images per id that is 10-20%, not 30%, and it is deterministic
   rather than random. Follow the paper and you build a harder dataset than the
   one behind the published numbers.
2. **Where the occluder lands.** The paper says instances are "randomly placed
   in horizontal directions". The code pins every class to a fixed corner --
   left_down, right_down, left_up, left_middle or right_middle. No randomness.

``select="one_per_id"`` + ``jitter=0.0`` reproduces the code (and therefore the
published numbers). ``select="ratio"`` + ``jitter>0`` implements the paper.
Which one you used belongs in the table caption either way.

Scale
-----
Each class has a fixed area ratio (``AREA_RATIO``): the instance is resized so
its bounding box covers that fraction of the crop, aspect ratio preserved.
``level=None`` keeps those ratios (MGCC behaviour). Passing a ``level`` overrides
all of them with one value, which is what puts ``ogor`` on the same "occlusion
coverage" x-axis as ``occlude_bench``'s blackout variants -- note the ratio is
of the instance's *bounding box*, so true pixel coverage is lower wherever the
instance is not convex.

The 60 instance PNGs (15 classes x 4 samples) live in the MGCC repo at
``dataset/occlusion_img/``; ``scripts/fetch_oil.py`` downloads them.
"""
from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image

log = logging.getLogger(__name__)

#: The 15 classes of MGCC's Occlusion Instance Library, verbatim from
#: ``Occlusion_Adding.occlusion_list``.
OCCLUSION_CLASSES: List[str] = [
    "bench", "bike", "car", "card", "chair", "firehydrant", "motorbike",
    "pedestrian", "post", "roadsign", "stone", "bag", "kite", "suitcase",
    "umbrella",
]

#: Fraction of the crop area covered by the instance's bounding box, verbatim
#: from ``Occlusion_Adding.area_ratio_dict``. Note the spread: a chair destroys
#: 65% of the image, a suitcase 10%. "Occluded-*" is therefore a *mixture* of
#: severities, not a single operating point -- which is why a Tier-2 sweep over
#: one coverage level is still worth having alongside it.
AREA_RATIO: Dict[str, float] = {
    "bench": 0.40, "bike": 0.60, "car": 0.45, "card": 0.20, "chair": 0.65,
    "firehydrant": 0.20, "motorbike": 0.50, "pedestrian": 0.50, "post": 0.50,
    "roadsign": 0.40, "stone": 0.15, "bag": 0.20, "kite": 0.40,
    "suitcase": 0.10, "umbrella": 0.25,
}

#: Anchor corner per class, verbatim from the if/elif chain in the released
#: ``__call__``. The paper's three "position prior" sets (Up / Middle / Bottom)
#: are these five anchors collapsed.
ANCHOR: Dict[str, str] = {
    **{c: "left_down" for c in
       ("stone", "motorbike", "bench", "bike", "car", "card", "chair", "post")},
    **{c: "right_down" for c in ("roadsign", "firehydrant", "pedestrian")},
    **{c: "left_up" for c in ("umbrella", "kite")},
    "bag": "left_middle",
    "suitcase": "right_middle",
}

#: Paper-side grouping, kept because it is what the paper's Table 1 shows and
#: what a reader will look for.
POSITION_SETS = {
    "up": ("umbrella", "kite"),
    "middle": ("bag", "suitcase", "post"),
    "bottom": ("car", "bike", "stone", "motorbike", "bench", "roadsign",
               "chair", "card", "pedestrian", "firehydrant"),
}

SAMPLES_PER_CLASS = 4


# --------------------------------------------------------------------------- #
# Instance library                                                             #
# --------------------------------------------------------------------------- #
class OIL:
    """The Occlusion Instance Library: 15 classes x 4 RGBA samples.

    Instances are loaded once and cached, because ``make_variant`` pastes onto
    tens of thousands of crops and re-decoding a PNG per crop dominates the
    runtime.
    """

    def __init__(self, root: str | Path, strict: bool = True):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"OIL not found at {self.root}. Run:\n"
                f"    python scripts/fetch_oil.py --out {self.root}")
        self._cache: Dict[str, Image.Image] = {}
        self.classes: List[str] = []
        missing = []
        for cls in OCCLUSION_CLASSES:
            have = [i for i in range(1, SAMPLES_PER_CLASS + 1)
                    if (self.root / f"{cls}{i}.png").is_file()]
            if len(have) == SAMPLES_PER_CLASS:
                self.classes.append(cls)
            else:
                missing.append(f"{cls} ({len(have)}/{SAMPLES_PER_CLASS})")
        if missing:
            msg = ("OIL is incomplete: " + ", ".join(missing) +
                   ". A partial library changes the occlusion distribution and "
                   "makes the numbers non-comparable to MGCC's.")
            if strict:
                raise FileNotFoundError(msg)
            log.warning("[ogor] %s", msg)
        if not self.classes:
            raise FileNotFoundError(f"no usable instances under {self.root}")

    def __len__(self) -> int:
        return len(self.classes) * SAMPLES_PER_CLASS

    def get(self, cls: str, i: int) -> Image.Image:
        key = f"{cls}{i}"
        img = self._cache.get(key)
        if img is None:
            img = Image.open(self.root / f"{key}.png").convert("RGBA")
            if not _has_alpha(img):
                # Three of MGCC's own 60 instances (card3, card4, bike4) ship
                # opaque, so they paste as solid rectangles rather than cut-out
                # objects. Their `generate_occlusion` never calls the `transPNG`
                # helper that would fix this, so the published Occluded-* sets
                # contain those rectangles too.
                #
                # Therefore: do NOT "fix" them when reproducing MGCC. Reported
                # once per instance because a *replacement* instance arriving
                # opaque would be a real error, and the log is the only place
                # that would show it.
                log.info("[ogor] %s.png is opaque -- its bounding box is "
                         "pasted whole. This matches MGCC (their transPNG "
                         "helper is never called), so leave it alone for the "
                         "reproduction; --whiten-to-alpha only if you are "
                         "building your own library.", key)
            self._cache[key] = img
        return img


def _has_alpha(img: Image.Image, thresh: int = 250) -> bool:
    """True if the alpha channel actually carries a cut-out (not all opaque)."""
    a = img.getchannel("A")
    lo, _hi = a.getextrema()
    return lo < thresh


# --------------------------------------------------------------------------- #
# The generator                                                                #
# --------------------------------------------------------------------------- #
def anchor_xy(anchor: str, W: int, H: int, w: int, h: int,
              jitter: float = 0.0, rng: random.Random | None = None) -> tuple:
    """Top-left paste coordinate for ``anchor``.

    Reproduces the released code exactly at ``jitter=0``. Note ``left_middle`` /
    ``right_middle`` place the instance's *bottom* edge on the midline (the code
    computes ``H/2 - h``), so a bag hangs above centre rather than straddling it.
    """
    if anchor == "left_down":
        x, y = 0, H - h
    elif anchor == "right_down":
        x, y = W - w, H - h
    elif anchor == "left_up":
        x, y = 0, 0
    elif anchor == "left_middle":
        x, y = 0, H // 2 - h
    elif anchor == "right_middle":
        x, y = W - w, H // 2 - h
    else:
        raise ValueError(f"unknown anchor {anchor!r}")
    if jitter and rng is not None:
        # the paper's "randomly place in horizontal directions"; the vertical
        # prior (what makes a car a car) is preserved
        span = jitter * max(W - w, 0)
        x = int(round(x + rng.uniform(-span, span)))
        x = max(min(x, W - w), 0) if w <= W else 0
    return x, y


def apply(img: Image.Image, oil: OIL, seed: int, level: float | None = None,
          jitter: float = 0.0, classes: Sequence[str] | None = None
          ) -> Image.Image:
    """Paste one random occlusion instance onto ``img``.

    Deterministic in ``seed``: the caller derives it from the image path, so a
    re-run -- or a different machine, or a different worker count -- reproduces
    byte-identical images. That is what lets every baseline be scored on the
    *same* occluded gallery.

    ``level`` overrides every class's area ratio with one value (for coverage
    sweeps); ``None`` keeps MGCC's per-class ratios.
    """
    rng = random.Random(seed)
    pool = list(classes) if classes else oil.classes
    cls = rng.choice(pool)
    i = rng.randint(1, SAMPLES_PER_CLASS)          # inclusive, as in MGCC

    inst = oil.get(cls, i)
    ratio = AREA_RATIO[cls] if level is None else float(level)
    ratio = max(min(ratio, 1.0), 0.0)
    if ratio <= 0:
        return img.copy()

    W, H = img.size
    wh = inst.size[0] / inst.size[1]
    area = ratio * W * H
    h = int(round(math.sqrt(area / wh)))
    w = int(round(math.sqrt(area * wh)))
    if w < 1 or h < 1:
        return img.copy()
    inst = inst.resize((w, h))

    out = img.convert("RGB").copy()
    xy = anchor_xy(ANCHOR[cls], W, H, w, h, jitter=jitter, rng=rng)
    out.paste(inst, xy, inst)                      # alpha-composited
    return out


# --------------------------------------------------------------------------- #
# Which images to occlude                                                      #
# --------------------------------------------------------------------------- #
def select(pids: Sequence[str], mode: str = "one_per_id", ratio: float = 0.3,
           seed: int = 0) -> List[bool]:
    """Boolean mask over a gallery: which entries get occluded.

    ``one_per_id`` -- MGCC's released code: the first image of each identity, in
    the order the gallery was built. Deterministic, no seed.

    ``ratio`` -- the paper's description: ``ratio`` of each identity's images,
    stratified per identity. Stratifying matters: sampling ``ratio`` globally
    would occlude *every* image of some identities and none of others, and an
    all-occluded identity is a different (much harder) retrieval problem than a
    mixed gallery, which is what both the paper and the code are after.

    ``all`` -- every image, for the Tier-2 sweeps where the question is "how far
    does this method fall at coverage X", not "does it survive a mixed gallery".
    """
    n = len(pids)
    if mode == "all":
        return [True] * n
    if mode == "one_per_id":
        seen, out = set(), []
        for p in pids:
            out.append(p not in seen)
            seen.add(p)
        return out
    if mode != "ratio":
        raise ValueError(f"unknown selection mode {mode!r}")

    by_pid: Dict[str, List[int]] = {}
    for i, p in enumerate(pids):
        by_pid.setdefault(str(p), []).append(i)
    mask = [False] * n
    for pid, idxs in by_pid.items():
        k = int(round(ratio * len(idxs)))
        if ratio > 0:
            k = max(k, 1) if len(idxs) else 0      # never silently skip an id
        k = min(k, len(idxs))
        # seeded by pid, not by position, so the choice survives a reordered or
        # subsetted gallery
        rng = random.Random(f"{seed}:{pid}")
        for i in rng.sample(idxs, k):
            mask[i] = True
    return mask
