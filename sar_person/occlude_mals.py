"""Generate occluded / degraded variants of MALS gene_crop for the
occlusion-aware retrieval study.

Motivation
----------
The paper targets *occlusion-aware* text-to-image person retrieval for
wilderness search-and-rescue: in the field a missing person is rarely a clean,
fully visible full-body crop -- they are partially hidden by foliage, lying in an
odd posture, seen in low light, or captured through a noisy drone/thermal sensor.
MALS ``gene_crop/4x`` gives clean, well-lit, full-body studio-style crops, so it
measures the *upper bound*. To probe robustness we synthesise controlled
degradations and re-run the same retrieval eval; the metric drop from clean ->
occluded is the occlusion-robustness signal.

Because MALS has no pose / keypoint annotations, body parts are approximated by
vertical thirds of the crop (head = top, torso = middle, legs = bottom) -- a
reasonable prior for upright full-body person crops.

Variants (any subset via --variants)
-------------------------------------
  head / torso / legs   : blackout the corresponding vertical third
  cutout                : N random rectangles covering ~--coverage of the image
  noise                 : additive Gaussian sensor noise (--severity)
  blur                  : Gaussian blur (out-of-focus / motion proxy)
  lowlight              : gamma-darken + mild noise (dusk / canopy shade)
  foliage               : procedural leaf + branch overlay (wilderness occluder)

Output layout (mirrors the MALS convention so eval_mals.py just works)
----------------------------------------------------------------------
  <out-root>/gene_crop_occ/<variant>/<stem>.jpg      # degraded images
  <out-root>/gene_attrs_occ/<variant>_attrs.json     # same caption+label, new path

The captions and 27-dim attribute labels are copied unchanged: the text query
still describes the *whole* person, only the image is degraded -- exactly the SAR
scenario (full verbal description, partially visible subject).

Example
-------
  python -m sar_person.occlude_mals \
      --attrs "../Dataset/MALS/gene_attrs/g_4x_attrs.json" \
      --out-root "../Dataset/MALS" \
      --variants torso cutout noise foliage \
      --limit 2000 --seed 0
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BODY_PARTS = {"head": (0.0, 0.33), "torso": (0.33, 0.66), "legs": (0.66, 1.0)}


# --------------------------------------------------------------------------- #
# Degradations (each takes/returns a PIL RGB image)                            #
# --------------------------------------------------------------------------- #
def blackout_third(img, part, fill=(0, 0, 0)):
    w, h = img.size
    lo, hi = BODY_PARTS[part]
    out = img.copy()
    ImageDraw.Draw(out).rectangle([0, int(lo * h), w, int(hi * h)], fill=fill)
    return out


def random_cutout(img, rng, coverage=0.35, n_boxes=3, fill=(0, 0, 0)):
    w, h = img.size
    out = img.copy()
    draw = ImageDraw.Draw(out)
    target = coverage * w * h
    done = 0.0
    for _ in range(n_boxes):
        bw = rng.uniform(0.2, 0.5) * w
        bh = (target / n_boxes) / max(bw, 1)
        bh = min(bh, h * 0.6)
        x0 = rng.uniform(0, max(w - bw, 1)); y0 = rng.uniform(0, max(h - bh, 1))
        draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=fill)
        done += bw * bh
        if done >= target:
            break
    return out


def gaussian_noise(img, rng, severity=25.0):
    a = np.asarray(img, np.float32)
    a += rng.normal(0, severity, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def gaussian_blur(img, radius=2.5):
    return img.filter(ImageFilter.GaussianBlur(radius))


def lowlight(img, rng, gamma=2.2, severity=8.0):
    a = np.asarray(img, np.float32) / 255.0
    a = np.power(a, gamma) * 255.0
    a += rng.normal(0, severity, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def foliage(img, rng, n_leaves=45, n_branches=6, coverage_bias=0.55):
    """Procedural wilderness occluder: translucent green/brown leaves + branches
    concentrated over part of the person (no external assets needed)."""
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # branches
    for _ in range(n_branches):
        x0 = rng.uniform(0, w); y0 = rng.uniform(0, h)
        x1 = x0 + rng.uniform(-0.5, 0.5) * w; y1 = y0 + rng.uniform(-0.5, 0.5) * h
        brown = (rng.randint(60, 110), rng.randint(40, 80), rng.randint(20, 50), 230)
        draw.line([x0, y0, x1, y1], fill=brown, width=rng.randint(3, 8))
    # leaves (ellipses) biased toward one vertical band for a realistic clump
    band_c = rng.uniform(0.2, 0.8) * h
    for _ in range(n_leaves):
        lw = rng.uniform(0.04, 0.13) * w
        lh = lw * rng.uniform(1.4, 2.6)
        cx = rng.uniform(0, w)
        cy = rng.gauss(band_c, coverage_bias * h * 0.25)
        cy = min(max(cy, 0), h)
        green = (rng.randint(20, 70), rng.randint(80, 150), rng.randint(20, 70),
                 rng.randint(180, 245))
        draw.ellipse([cx - lw, cy - lh, cx + lw, cy + lh], fill=green)
    out = img.convert("RGBA")
    out.alpha_composite(overlay)
    return out.convert("RGB")


def apply_variant(img, variant, rng, args):
    if variant in BODY_PARTS:
        return blackout_third(img, variant, fill=tuple(args.fill))
    if variant == "cutout":
        return random_cutout(img, rng, coverage=args.coverage, fill=tuple(args.fill))
    if variant == "noise":
        return gaussian_noise(img, rng, severity=args.severity)
    if variant == "blur":
        return gaussian_blur(img, radius=args.blur_radius)
    if variant == "lowlight":
        return lowlight(img, rng, severity=args.severity)
    if variant == "foliage":
        return foliage(img, rng)
    raise ValueError(f"unknown variant {variant!r}")


ALL_VARIANTS = ["head", "torso", "legs", "cutout", "noise", "blur", "lowlight", "foliage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", required=True)
    ap.add_argument("--mals-root", default=None,
                    help="root the attrs image paths are relative to (default: MALS/)")
    ap.add_argument("--out-root", default=None,
                    help="where gene_crop_occ/ + gene_attrs_occ/ are written "
                         "(default: same as --mals-root)")
    ap.add_argument("--variants", nargs="+", default=["torso", "cutout", "noise", "foliage"],
                    choices=ALL_VARIANTS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coverage", type=float, default=0.35, help="cutout area fraction")
    ap.add_argument("--severity", type=float, default=25.0, help="noise sigma (0-255)")
    ap.add_argument("--blur-radius", type=float, default=2.5)
    ap.add_argument("--fill", type=int, nargs=3, default=[0, 0, 0], help="blackout RGB")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    attrs_path = Path(args.attrs).resolve()
    mals_root = Path(args.mals_root).resolve() if args.mals_root else attrs_path.parents[1]
    out_root = Path(args.out_root).resolve() if args.out_root else mals_root
    records = json.loads(attrs_path.read_text(encoding="utf-8"))
    if args.stride > 1:
        records = records[::args.stride]
    if args.limit is not None and args.limit < len(records):
        records = random.Random(args.seed).sample(records, args.limit)

    img_root = out_root / "gene_crop_occ"
    attr_root = out_root / "gene_attrs_occ"
    attr_root.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        vdir = img_root / variant
        vdir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(f"{args.seed}:{variant}")
        nprng = np.random.default_rng(abs(hash((args.seed, variant))) % (2**32))
        new_records, kept = [], 0
        for rec in records:
            rel = rec["image"]
            src = rel if os.path.isabs(rel) else str(mals_root / rel)
            if not os.path.exists(src):
                continue
            stem = Path(rel).stem + ".jpg"
            dst = vdir / stem
            try:
                img = Image.open(src).convert("RGB")
            except Exception:
                continue
            # feed both rng streams (PIL-side uses `rng`, numpy noise uses nprng)
            _rng = _DualRNG(rng, nprng)
            out = apply_variant(img, variant, _rng, args)
            out.save(dst, quality=args.quality)
            new_rel = os.path.relpath(dst, out_root)
            nr = dict(rec)
            nr["image"] = new_rel.replace(os.sep, "/")
            nr["image_id"] = f"{variant}_{rec.get('image_id', stem)}"
            new_records.append(nr)
            kept += 1
        out_json = attr_root / f"{variant}_attrs.json"
        out_json.write_text(json.dumps(new_records, indent=0))
        print(f"[{variant}] wrote {kept} images -> {vdir}  ({out_json.name})")


class _DualRNG:
    """Adapter so degradations can call rng.uniform/randint/gauss (python
    random) and rng.normal (numpy) through one object."""
    def __init__(self, pyrng, nprng):
        self._py = pyrng; self._np = nprng
    def uniform(self, a, b): return self._py.uniform(a, b)
    def randint(self, a, b): return self._py.randint(a, b)
    def gauss(self, m, s): return self._py.gauss(m, s)
    def normal(self, loc, scale, size): return self._np.normal(loc, scale, size)


if __name__ == "__main__":
    main()
