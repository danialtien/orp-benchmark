#!/usr/bin/env python3
"""Check that YOUR regenerated ORP occlusions are the ones used in the paper.

The ORP galleries are not redistributed (they are derived from licensed
images). Instead, ``manifests/occluded/<bench>_<geometry>_lNN.json.gz`` hold a
fixed spot-check sample of gallery images for every one of the 15 cells, each
with
    * ``seed``        the per-image occluder seed (sha1 of the dataset-relative
                      image id, see sar_person.occlude_bench._seed_for),
    * ``rgb_sha256``  SHA-256 of the occluded image as a raw RGB array, BEFORE
                      JPEG encoding  -> depends only on the source pixels and
                      the occluder code (portable across machines),
    * ``jpg_sha256``  SHA-256 of the JPEG file (quality 95) -> additionally
                      depends on the libjpeg build inside Pillow,
    * ``changed``     fraction of pixels the occluder modified.

    # verify (uses your local dataset, writes nothing to disk)
    python scripts/verify_occlusion.py --data-root ../Dataset --benchmarks cuhk rstp icfg

    # maintainers: (re)write the reference sample
    python scripts/verify_occlusion.py --data-root ../Dataset --write-reference

``rgb_sha256`` must match for every sampled image. A ``jpg_sha256`` mismatch
with matching ``rgb_sha256`` only means a different Pillow/libjpeg build; it
does not change which pixels were occluded.

Needs: numpy, Pillow (no torch, no GPU).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sar_person.occlude_bench import degrade, _seed_for  # noqa: E402
from sar_person.eval_sets import build as build_evalset  # noqa: E402

GEOMS = ("torso", "cutout", "foliage")
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5)
N_SAMPLE = {"cuhk": 300, "rstp": 300, "icfg": 300}
MAN = REPO / "manifests"


def load_gz(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def dump_gz(obj, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def render(src: Path, image_id: str, geom: str, level: float):
    img = Image.open(src).convert("RGB")
    out = degrade(img, geom, level, _seed_for(image_id))
    a = np.asarray(out, dtype=np.uint8)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=95)
    changed = float((a != np.asarray(img, dtype=np.uint8)).any(-1).mean())
    return (hashlib.sha256(a.tobytes()).hexdigest(), hashlib.sha256(buf.getvalue()).hexdigest(),
            round(changed, 6), list(out.size))


#: where each benchmark's gallery images live under --data-root, keyed by the
#: dataset-relative image id stored in manifests/evalsets/*.json.gz
IMAGE_PATH = {
    "cuhk": lambda root, iid: root / "CUHK-PEDES" / "imgs" / iid,
    "icfg": lambda root, iid: root / "ICFG-PEDES" / "ICFG-PEDES" / "imgs" / iid,
    # images extracted from the RSTPReid parquet by sar_person.eval_sets, or the
    # official release's imgs/ folder (scripts/prepare_rstp_official.py)
    "rstp": lambda root, iid: next(
        (p for p in (root / "RSTPReid" / "images" / f"test-00000-of-00001__{Path(iid).name}",
                     root / "RSTPReid" / "imgs" / Path(iid).name) if p.exists()),
        root / "RSTPReid" / "imgs" / Path(iid).name),
}


class _Gallery:
    def __init__(self, gallery):
        self.gallery = gallery


def local_gallery(bench, data_root, evalset_cache, from_manifest=False):
    """Gallery in evaluation order. By default it is rebuilt from your
    annotation files with sar_person.eval_sets.build (the paper's code path);
    with --from-manifest (or if that fails, e.g. pyarrow missing for RSTPReid)
    the released id list is resolved against --data-root instead."""
    ref = load_gz(MAN / "evalsets" / f"{bench}.json.gz")
    if not from_manifest:
        try:
            es = build_evalset(bench, data_root, cache_dir=evalset_cache)
            if [g["image_id"] for g in es.gallery] != [g[0] for g in ref["gallery"]]:
                raise SystemExit(f"[{bench}] your local gallery differs from the released evaluation set "
                                 f"(order or membership) -- run scripts/verify_evalsets.py first.")
            return es
        except (ImportError, FileNotFoundError) as e:
            print(f"[{bench}] eval_sets.build unavailable ({type(e).__name__}: {e}); "
                  f"resolving the released id list instead")
    root = Path(data_root).expanduser().resolve()
    gal = [{"image": str(IMAGE_PATH[bench](root, iid)), "image_id": iid, "pid": pid}
           for iid, pid in ref["gallery"]]
    return _Gallery(gal)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=str(REPO.parent / "Dataset"))
    ap.add_argument("--evalset-cache", default=None,
                    help="directory with cached evaluation sets (optional)")
    ap.add_argument("--benchmarks", nargs="+", default=["cuhk", "rstp", "icfg"])
    ap.add_argument("--cells", nargs="*", default=None, help="e.g. foliage_50 (default: all 15)")
    ap.add_argument("--from-manifest", action="store_true",
                    help="resolve the released gallery id list against --data-root instead of "
                         "rebuilding it from the annotation files")
    ap.add_argument("--write-reference", action="store_true")
    a = ap.parse_args()
    all_ok = True
    for b in a.benchmarks:
        es = local_gallery(b, a.data_root, a.evalset_cache, a.from_manifest)
        n = len(es.gallery)
        idx = sorted(np.random.default_rng(0).choice(n, size=min(N_SAMPLE[b], n), replace=False).tolist())
        for g in GEOMS:
            for lv in LEVELS:
                cell = f"{g}_{int(round(lv * 100)):02d}"
                if a.cells and cell not in a.cells:
                    continue
                p = MAN / "occluded" / f"{b}_{g}_l{int(round(lv * 100)):02d}.json.gz"
                t0 = time.time()
                rows = []
                for i in idx:
                    gi = es.gallery[i]
                    rgb, jpg, ch, size = render(Path(gi["image"]), gi["image_id"], g, lv)
                    rows.append([i, gi["image_id"], _seed_for(gi["image_id"]), rgb, jpg, ch, size])
                if a.write_reference:
                    from PIL import __version__ as pil_v
                    dump_gz({"benchmark": b, "geometry": g, "nominal_coverage": lv,
                             "generator": "sar_person.occlude_bench.degrade", "jpeg_quality": 95,
                             "reference_env": {"pillow": pil_v, "numpy": np.__version__,
                                               "python": sys.version.split()[0]},
                             "sample": "numpy.random.default_rng(0).choice(n_gallery, %d, replace=False)" % len(idx),
                             "columns": ["gallery_index", "image_id", "seed", "rgb_sha256", "jpg_sha256",
                                         "changed_fraction", "size_wh"],
                             "rows": rows}, p)
                    print(f"[ref] {b} {cell}: {len(rows)} images, mean changed {np.mean([r[5] for r in rows]):.3f} "
                          f"({time.time() - t0:.0f}s)")
                    continue
                ref = load_gz(p)
                R = {r[0]: r for r in ref["rows"]}
                n_rgb = sum(R[r[0]][3] == r[3] for r in rows)
                n_jpg = sum(R[r[0]][4] == r[4] for r in rows)
                ok = n_rgb == len(rows)
                all_ok &= ok
                print(f"{b:5s} {cell:11s} pixels identical {n_rgb}/{len(rows)}  jpeg identical {n_jpg}/{len(rows)}"
                      f"  {'OK' if ok else 'MISMATCH'}")
    if not a.write_reference:
        print("ALL CELLS MATCH" if all_ok else "SOME CELLS DIFFER -- see above")
        return 0 if all_ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
