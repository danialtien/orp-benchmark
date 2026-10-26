"""Tier-2: build occluded/degraded variants of a full retrieval benchmark.

``occlude_mals`` degrades a MALS caption file. This module degrades an
:class:`~sar_person.eval_sets.EvalSet`, which is what the paper actually needs:
the *test gallery* of ICFG-PEDES / RSTPReid / CUHK-PEDES is degraded while the
queries, the pids and the gallery size stay byte-identical, so the drop from
clean -> occluded is attributable to the degradation and nothing else.

Two axes (the third, "which benchmark", is the caller's choice):

  variant : head | torso | legs | cutout | foliage | lowlight | noise | blur
  level   : 0..1 severity. For the body-part and cutout variants this is the
            fraction of the crop that is destroyed, so `level` is directly the
            "occlusion coverage" the paper plots against; POP reports real
            foliage occlusion up to ~70%, hence the default 0.25/0.50/0.70 grid.

Every degraded image is written once and reused; a manifest per (variant, level)
lands next to them, ready for ``eval_protocol --manifest``.

    python -m sar_person.occlude_bench --benchmark rstp \
        --variants torso cutout foliage lowlight --levels 0.25 0.5 0.7 \
        --out-root ../Dataset/Result/occluded --workers 8

    # then, for each manifest:
    python -m sar_person.eval_protocol --manifest .../rstp_torso_l50.json ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from functools import partial
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from . import ogor as _ogor
from .eval_sets import EvalSet, build as build_evalset
from .occlude_mals import foliage as _foliage

log = logging.getLogger(__name__)

BODY_PARTS = {"head": 0.165, "torso": 0.5, "legs": 0.835}   # vertical centre
VARIANTS = ["head", "torso", "legs", "cutout", "foliage", "lowlight",
            "noise", "blur", "ogor"]
#: variants whose `level` is a true area fraction (plottable as "coverage")
COVERAGE_VARIANTS = {"head", "torso", "legs", "cutout", "ogor"}
#: variants that need the instance library on disk
OIL_VARIANTS = {"ogor"}

# One OIL per worker process. The library holds decoded RGBA images, so it is
# both expensive to build (60 PNG decodes) and expensive to pickle -- passing it
# through Pool.map would ship it once per chunk. Workers build their own on
# first use instead.
_OIL_CACHE: Dict[str, "_ogor.OIL"] = {}


def get_oil(root, strict: bool = True) -> "_ogor.OIL":
    key = f"{root}|{strict}"
    if key not in _OIL_CACHE:
        _OIL_CACHE[key] = _ogor.OIL(root, strict=strict)
    return _OIL_CACHE[key]


# --------------------------------------------------------------------------- #
# Degradations, all parameterised by a single 0..1 level                       #
# --------------------------------------------------------------------------- #
def blackout_band(img: Image.Image, part: str, level: float,
                  fill=(0, 0, 0)) -> Image.Image:
    """Destroy a horizontal band of height ``level * h`` centred on ``part``.

    ``occlude_mals.blackout_third`` always removes exactly one third, which
    makes "coverage" un-sweepable. Centring a variable-height band on the part's
    centre keeps the semantics (head/torso/legs) while making level meaningful.
    """
    w, h = img.size
    c = BODY_PARTS[part] * h
    half = max(1.0, level * h / 2)
    y0, y1 = max(0.0, c - half), min(float(h), c + half)
    out = img.copy()
    ImageDraw.Draw(out).rectangle([0, y0, w, y1], fill=fill)
    return out


def cutout(img: Image.Image, level: float, rng: random.Random,
           n_boxes: int = 3, fill=(0, 0, 0)) -> Image.Image:
    """``n_boxes`` random rectangles covering ~``level`` of the area."""
    w, h = img.size
    out = img.copy()
    draw = ImageDraw.Draw(out)
    target, done = level * w * h, 0.0
    for _ in range(n_boxes):
        bw = rng.uniform(0.25, 0.6) * w
        bh = min((target - done) / max(bw, 1), h * 0.7)
        if bh <= 1:
            break
        x0 = rng.uniform(0, max(w - bw, 1))
        y0 = rng.uniform(0, max(h - bh, 1))
        draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=fill)
        done += bw * bh
        if done >= target:
            break
    return out


def lowlight(img: Image.Image, level: float, rng: np.random.Generator) -> Image.Image:
    """Dusk / canopy shade: gamma-darken plus a little sensor noise."""
    gamma = 1.0 + 2.5 * level                       # level 0.7 -> gamma 2.75
    a = np.asarray(img, np.float32) / 255.0
    a = np.power(a, gamma) * 255.0
    a += rng.normal(0, 4.0 + 8.0 * level, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def noise(img: Image.Image, level: float, rng: np.random.Generator) -> Image.Image:
    a = np.asarray(img, np.float32) + rng.normal(0, 45.0 * level, (*img.size[::-1], 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def blur(img: Image.Image, level: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(0.5 + 4.0 * level))


def degrade(img: Image.Image, variant: str, level: float, seed: int,
            oil_root=None, oil_strict: bool = True, jitter: float = 0.0,
            ogor_native_scale: bool = True) -> Image.Image:
    """Deterministic given (variant, level, seed) -- the seed is derived from the
    image path by the caller, so re-running reproduces byte-identical images.

    ``ogor_native_scale`` keeps MGCC's per-class area ratios (what reproduces
    their dataset) and ignores ``level``; set it False to force every class to
    ``level`` so ogor joins the other coverage variants on one sweep axis.
    """
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    if variant in BODY_PARTS:
        return blackout_band(img, variant, level)
    if variant == "ogor":
        if oil_root is None:
            raise ValueError("variant 'ogor' needs --oil-root "
                             "(python scripts/fetch_oil.py)")
        return _ogor.apply(img, get_oil(oil_root, oil_strict), seed,
                           level=None if ogor_native_scale else level,
                           jitter=jitter)
    if variant == "cutout":
        return cutout(img, level, py_rng)
    if variant == "foliage":
        return _foliage(img, py_rng, n_leaves=int(20 + 90 * level),
                        n_branches=int(3 + 10 * level))
    if variant == "lowlight":
        return lowlight(img, level, np_rng)
    if variant == "noise":
        return noise(img, level, np_rng)
    if variant == "blur":
        return blur(img, level)
    raise ValueError(f"unknown variant {variant!r}")


# --------------------------------------------------------------------------- #
# Batch generation                                                             #
# --------------------------------------------------------------------------- #
def _seed_for(key: str) -> int:
    """Per-image seed from a *dataset-relative* key, via sha1.

    Two properties are load-bearing and neither is optional:

    **Portable.** The key is the gallery entry's ``image_id`` -- the path
    relative to the dataset root -- not the absolute path. Generating a gallery
    on a Windows workstation and on a Linux server must produce byte-identical
    images, or the "same occluded gallery for every baseline" guarantee only
    holds per machine.

    **Well mixed.** The previous version hashed the last 8 bytes of the absolute
    path. On ICFG those bytes are the frame-index tail of the filename
    (``..._0015_0.jpg`` -> ``15_0.jpg``), which yields **710 distinct seeds for
    19 848 images** and skews the occluder class distribution **5.8x** -- one
    class drawn 4 029 times, another 697, where MGCC's ``random.choice`` is
    uniform at 1 323. A reproduction with the wrong occlusion mix fails the
    57.24 gate check for reasons that have nothing to do with the pipeline.
    sha1 over the full relative key gives 19 848 distinct seeds and 1.11x skew.
    """
    return int.from_bytes(hashlib.sha1(str(key).encode("utf-8")).digest()[:4],
                          "little") & 0x7FFFFFFF


def _one(job, variant: str, level: float, quality: int, **kw):
    src, dst, key = job
    dst = Path(dst)
    if dst.exists():
        return src, str(dst)
    try:
        img = Image.open(src).convert("RGB")
    except (OSError, ValueError) as e:
        log.warning("[occ] cannot read %s (%s)", src, e)
        return src, None
    dst.parent.mkdir(parents=True, exist_ok=True)
    degrade(img, variant, level, _seed_for(key), **kw).save(dst, quality=quality)
    return src, str(dst)


def make_variant(es: EvalSet, variant: str, level: float, out_root: Path,
                 workers: int = 0, quality: int = 95,
                 select: str = "all", occ_ratio: float = 0.3,
                 select_seed: int = 0, oil_root=None, oil_strict: bool = True,
                 jitter: float = 0.0, ogor_native_scale: bool = True) -> EvalSet:
    """Degrade the gallery of ``es`` and return the derived EvalSet.

    ``select`` decides *which* gallery images are degraded (see
    ``ogor.select``). Anything not selected is carried over at its original
    path -- not copied, not re-encoded. Two reasons that matters: the gallery
    size and pid distribution stay byte-identical to the clean set, so a
    clean-vs-occluded delta is attributable to the occlusion alone; and JPEG
    round-tripping 70% of a gallery for no reason would itself shift R@1.
    """
    lvl = int(round(level * 100))
    tag = f"{es.name}_{variant}_l{lvl:02d}"
    if select != "all":
        tag += f"_{select}" + (f"{occ_ratio:g}" if select == "ratio" else "")
    # absolute, for the same reason eval_sets.build resolves data_root
    img_dir = Path(out_root).expanduser().resolve() / "images" / tag

    mask = _ogor.select([g["pid"] for g in es.gallery], mode=select,
                        ratio=occ_ratio, seed=select_seed)
    jobs = [(g["image"], str(img_dir / f"{i:07d}_{Path(g['image']).stem}.jpg"),
             g.get("image_id") or Path(g["image"]).name)
            for i, g in enumerate(es.gallery) if mask[i]]

    fn = partial(_one, variant=variant, level=level, quality=quality,
                 oil_root=oil_root, oil_strict=oil_strict, jitter=jitter,
                 ogor_native_scale=ogor_native_scale)
    if workers and workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            results = pool.map(fn, jobs, chunksize=32)
    else:
        results = [fn(j) for j in jobs]

    mapping = {s: d for s, d in results if d}
    # identity mapping for the untouched images, so with_gallery_images keeps
    # them instead of dropping them out of the gallery
    for i, g in enumerate(es.gallery):
        if not mask[i]:
            mapping.setdefault(g["image"], g["image"])

    n_occ = sum(mask)
    derived = es.with_gallery_images(mapping, tag, {
        "variant": variant, "level": level,
        "is_coverage": variant in COVERAGE_VARIANTS,
        "clean_manifest": es.name,
        "select": select, "occ_ratio": occ_ratio if select == "ratio" else None,
        "n_occluded": n_occ, "n_gallery": len(es.gallery),
        "occluded_frac": round(n_occ / max(len(es.gallery), 1), 4),
        "ogor_native_scale": ogor_native_scale if variant == "ogor" else None,
        "jitter": jitter if variant == "ogor" else None,
    })
    log.info("[occ] %s: %d/%d gallery image(s), %d occluded (%.1f%%)",
             tag, len(derived.gallery), len(es.gallery), n_occ,
             100 * n_occ / max(len(es.gallery), 1))
    return derived


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", help="icfg | rstp | cuhk | nomad")
    ap.add_argument("--manifest", help="clean EvalSet json (instead of --benchmark)")
    ap.add_argument("--data-root", default=str(
        Path(__file__).resolve().parents[2] / "Dataset"))
    ap.add_argument("--evalset-cache", default=None)
    ap.add_argument("--variants", nargs="+", default=["torso", "cutout",
                                                      "foliage", "lowlight"],
                    choices=VARIANTS)
    ap.add_argument("--levels", nargs="+", type=float, default=[0.25, 0.5, 0.7])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--workers", type=int, default=0,
                    help="processes for image writing (0/1 = serial)")
    ap.add_argument("--quality", type=int, default=95)
    # --- OGor / MGCC reproduction ---------------------------------------- #
    ap.add_argument("--oil-root", default=str(
        Path(__file__).resolve().parents[2] / "Dataset" / "OIL"),
        help="occlusion instance library (scripts/fetch_oil.py)")
    ap.add_argument("--oil-not-strict", action="store_true",
                    help="allow an incomplete OIL -- changes the occlusion "
                         "distribution, so the numbers stop being comparable "
                         "to MGCC's")
    ap.add_argument("--occ-select", default="all",
                    choices=["all", "one_per_id", "ratio"],
                    help="which gallery images to occlude. 'one_per_id' "
                         "reproduces MGCC's released code; 'ratio' + "
                         "--occ-ratio 0.3 implements their paper's text; "
                         "'all' is the Tier-2 coverage sweep (default)")
    ap.add_argument("--occ-ratio", type=float, default=0.3,
                    help="fraction of each identity's images to occlude, "
                         "used only with --occ-select ratio")
    ap.add_argument("--select-seed", type=int, default=0)
    ap.add_argument("--ogor-jitter", type=float, default=0.0,
                    help="random horizontal displacement as a fraction of the "
                         "free width. 0 = MGCC's fixed corners (default); the "
                         "paper describes ~0.5")
    ap.add_argument("--ogor-sweep-scale", action="store_true",
                    help="force every occluder to cover --levels instead of "
                         "MGCC's per-class ratios; puts ogor on the same "
                         "coverage axis as torso/cutout, and off MGCC's")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug only -- shrinks the gallery and makes the "
                         "numbers non-comparable")
    ap.add_argument("--cuhk-json", default=None)
    ap.add_argument("--cuhk-images", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    if bool(args.benchmark) == bool(args.manifest):
        raise SystemExit("pass exactly one of --benchmark / --manifest")
    es = (EvalSet.load(args.manifest) if args.manifest else
          build_evalset(args.benchmark, args.data_root,
                        cache_dir=args.evalset_cache,
                        cuhk_json=args.cuhk_json, cuhk_images=args.cuhk_images))
    if args.limit:
        log.warning("[occ] --limit %d: DEBUG ONLY, results are not comparable",
                    args.limit)
        es = EvalSet(es.name, es.gallery[:args.limit], es.queries, es.meta)

    native = not args.ogor_sweep_scale
    if "ogor" in args.variants:
        # fail before writing 20k images, not after
        oil = get_oil(args.oil_root, strict=not args.oil_not_strict)
        log.info("[occ] OIL: %d instance(s) over %d class(es) from %s",
                 len(oil), len(oil.classes), args.oil_root)
        if native and len(args.levels) > 1:
            log.warning("[occ] ogor uses MGCC's per-class area ratios, so all "
                        "%d --levels produce the SAME images (only the tag "
                        "differs). Pass --ogor-sweep-scale to sweep coverage.",
                        len(args.levels))

    out_root = Path(args.out_root)
    man_dir = out_root / "manifests"
    written: List[Dict] = []
    for variant in args.variants:
        for level in args.levels:
            # severity-only variants have no meaningful "coverage" reading, but
            # sweeping them on the same grid keeps the plots on one x-axis
            d = make_variant(es, variant, level, out_root, workers=args.workers,
                             quality=args.quality, select=args.occ_select,
                             occ_ratio=args.occ_ratio,
                             select_seed=args.select_seed,
                             oil_root=args.oil_root,
                             oil_strict=not args.oil_not_strict,
                             jitter=args.ogor_jitter,
                             ogor_native_scale=native)
            p = d.save(man_dir / f"{d.name}.json")
            written.append({"variant": variant, "level": level,
                            "manifest": str(p), "gallery": len(d.gallery),
                            "n_occluded": d.meta.get("n_occluded")})
            if variant == "ogor" and native:
                break                      # one level is all ogor can produce

    idx = out_root / f"index_{es.name}.json"
    idx.write_text(json.dumps({"clean": es.name, "variants": written}, indent=2),
                   encoding="utf-8")
    print(json.dumps({"index": str(idx), "n_manifests": len(written)}, indent=2))


if __name__ == "__main__":
    main()
