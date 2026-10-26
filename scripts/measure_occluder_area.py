"""Measure how much of a person crop each occluder ACTUALLY destroys.

The nominal `--levels 0.3` is a knob, not a measurement: a contiguous torso band
at 0.3 and scattered foliage at 0.3 do not cover the same number of pixels, so a
table indexed by nominal level compares two different things. This script paints
each occluder on real ICFG test crops and reports the measured destroyed area,
the fraction of ViT patches touched, and the size of the largest CLEAN region --
which is what actually survives to the encoder.

    python scripts/measure_occluder_area.py                       # print
    python scripts/measure_occluder_area.py --out area.csv        # + CSV
    python scripts/measure_occluder_area.py \
        --results ../Dataset/Result/occ_study/tables/occlusion_table.csv

With --results the mINP-retained column is read from the sweep you actually ran.
Without it the script falls back to a table hard-coded on 2026-08-20 and says so
loudly: publishing that snapshot next to a newer sweep is how a paper ends up
with two different numbers for one experiment.
"""
import argparse
import csv
import hashlib
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy import ndimage
except ImportError:                                   # optional
    ndimage = None

H, W, P = 384, 128, 16
BODY = {"head": 0.165, "torso": 0.5, "legs": 0.835}
VARIANTS = ["torso", "cutout", "foliage"]
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]

#: mINP retained on ICFG, snapshot of the 2026-08-20 sweep. Fallback only.
_MINP_SNAPSHOT = {
    ("torso", .1): 81.5, ("torso", .2): 67.4, ("torso", .3): 52.8,
    ("torso", .4): 37.8, ("torso", .5): 22.4,
    ("cutout", .1): 61.5, ("cutout", .2): 37.3, ("cutout", .3): 18.2,
    ("cutout", .4): 9.7, ("cutout", .5): 5.9,
    ("foliage", .1): 11.3, ("foliage", .2): 7.1, ("foliage", .3): 5.3,
    ("foliage", .4): 4.6, ("foliage", .5): 3.3,
}


def seed_for(k):
    return int.from_bytes(hashlib.sha1(str(k).encode()).digest()[:4],
                          "little") & 0x7FFFFFFF


def foliage_ov(size, rng, nl, nb, cb=0.55):
    w, h = size
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for _ in range(nb):
        x0 = rng.uniform(0, w); y0 = rng.uniform(0, h)
        x1 = x0 + rng.uniform(-.5, .5) * w
        y1 = y0 + rng.uniform(-.5, .5) * h
        d.line([x0, y0, x1, y1],
               fill=(rng.randint(60, 110), rng.randint(40, 80),
                     rng.randint(20, 50), 230), width=rng.randint(3, 8))
    bc = rng.uniform(.2, .8) * h
    for _ in range(nl):
        lw = rng.uniform(.04, .13) * w
        lh = lw * rng.uniform(1.4, 2.6)
        cx = rng.uniform(0, w)
        cy = min(max(rng.gauss(bc, cb * h * .25), 0), h)
        d.ellipse([cx - lw, cy - lh, cx + lw, cy + lh],
                  fill=(rng.randint(20, 70), rng.randint(80, 150),
                        rng.randint(20, 70), rng.randint(180, 245)))
    return ov.split()[3]


def torso_m(size, level):
    w, h = size
    m = Image.new("L", (w, h), 0)
    c = BODY["torso"] * h
    half = max(1., level * h / 2)
    ImageDraw.Draw(m).rectangle([0, max(0., c - half), w,
                                min(float(h), c + half)], fill=255)
    return m


def cutout_m(size, level, rng, n_boxes=3):
    w, h = size
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    target, done = level * w * h, 0.
    for _ in range(n_boxes):
        bw = rng.uniform(.25, .6) * w
        bh = min((target - done) / max(bw, 1), h * .7)
        if bh <= 1:
            break
        x0 = rng.uniform(0, max(w - bw, 1))
        y0 = rng.uniform(0, max(h - bh, 1))
        d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=255)
        done += bw * bh
        if done >= target:
            break
    return m


def stats(maskL):
    a = np.asarray(maskL, np.float32) / 255.
    area = float((a > 0).mean())
    m = np.asarray(maskL.resize((W, H), Image.BICUBIC), np.float32) / 255.
    pg = m.reshape(H // P, P, W // P, P).transpose(0, 2, 1, 3) \
          .reshape(H // P, W // P, P * P)
    hit = pg.max(axis=2) > 0.02                       # patch touched at all
    clean = ~hit
    if ndimage is None:
        return area, float(hit.mean()), float("nan"), -1
    lab, n = ndimage.label(clean)
    largest = 0 if n == 0 else max((lab == i).sum() for i in range(1, n + 1))
    return area, float(hit.mean()), float(largest / clean.size), int(n)


def minp_from_results(path: Path, benchmark="icfg", model=None):
    """mINP retained per (variant, level), read from occlusion_table.csv."""
    out = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("benchmark") != benchmark or row.get("variant") == "clean":
                continue
            if model and row.get("model") != model:
                continue
            try:
                lv = round(float(row["level"]), 2)
                ret = float(row["ret_mINP"])
            except (KeyError, TypeError, ValueError):
                continue
            out.setdefault((row["variant"], lv), []).append(100 * ret)
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(here.parents[2]), metavar="DIR",
                    help="project root that holds Dataset/ (default: two "
                         "levels above this script)")
    ap.add_argument("--imgdir", default=None, metavar="DIR",
                    help="override the crop directory outright")
    ap.add_argument("--n", type=int, default=150,
                    help="crops sampled per (variant, level); default 150")
    ap.add_argument("--out", default=None, metavar="CSV",
                    help="also write the measured areas here")
    ap.add_argument("--results", default=None, metavar="CSV",
                    help="occlusion_table.csv to read the REAL mINP retention "
                         "from, instead of the 2026-08-20 snapshot")
    ap.add_argument("--model", default=None,
                    help="with --results, restrict to one model")
    args = ap.parse_args()

    imgdir = Path(args.imgdir) if args.imgdir else \
        Path(args.root) / "Dataset/ICFG-PEDES/ICFG-PEDES/imgs/test"
    if not imgdir.is_dir():
        raise SystemExit(
            f"crop directory not found: {imgdir}\n"
            f"pass --root <project root holding Dataset/> or --imgdir <dir>")

    paths = sorted(imgdir.rglob("*.jpg"))[:1200][::3][:400]
    if not paths:
        raise SystemExit(f"no .jpg under {imgdir}")
    print(f"{len(paths)} real ICFG test crops from {imgdir}")
    if ndimage is None:
        print("NOTE: scipy missing -> 'largest CLEAN blob' unavailable "
              "(pip install scipy)")
    print()
    print(f"{'variant':8}{'lvl':>5}{'destroyed area':>16}{'patches hit':>13}"
          f"{'largest CLEAN blob':>20}{'#clean blobs':>14}")
    print("-" * 78)

    res, rows = {}, []
    for var in VARIANTS:
        for lv in LEVELS:
            A, Hh, L, N = [], [], [], []
            for p in paths[:args.n]:
                with Image.open(p) as im:
                    size = im.size
                rng = random.Random(seed_for(p.name))
                if var == "torso":
                    mk = torso_m(size, lv)
                elif var == "cutout":
                    mk = cutout_m(size, lv, rng)
                else:
                    mk = foliage_ov(size, rng, int(20 + 90 * lv),
                                    int(3 + 10 * lv))
                a, h_, l_, n_ = stats(mk)
                A.append(a); Hh.append(h_); L.append(l_); N.append(n_)
            res[(var, lv)] = (np.mean(A), np.mean(Hh), np.mean(L), np.mean(N))
            rows.append({"variant": var, "level": lv,
                         "measured_area": float(np.mean(A)),
                         "patches_hit": float(np.mean(Hh)),
                         "largest_clean_blob": float(np.mean(L)),
                         "n_clean_blobs": float(np.mean(N))})
            print(f"{var:8}{lv:5.1f}{100 * np.mean(A):15.1f}%"
                  f"{100 * np.mean(Hh):12.1f}%{100 * np.mean(L):19.1f}%"
                  f"{np.mean(N):14.1f}")

    minp, src = _MINP_SNAPSHOT, "2026-08-20 SNAPSHOT (hard-coded)"
    if args.results:
        got = minp_from_results(Path(args.results), model=args.model)
        if got:
            minp, src = got, str(args.results)
        else:
            print(f"\n!! no usable rows in {args.results} -- falling back")
    print(f"\n=== mINP retained vs MEASURED destroyed area ===")
    print(f"source of mINP retained: {src}")
    if minp is _MINP_SNAPSHOT:
        print("!! WARNING: these mINP numbers are a SNAPSHOT, not this sweep.")
        print("!! Re-run with --results .../occlusion_table.csv before quoting.")
    print(f"{'variant':8}{'lvl':>5}{'measured area':>15}{'mINP kept':>11}")
    print("-" * 40)
    for var in VARIANTS:
        for lv in LEVELS:
            v = minp.get((var, lv))
            for r in rows:
                if r["variant"] == var and r["level"] == lv:
                    r["minp_retained_pct"] = v
            print(f"{var:8}{lv:5.1f}{100 * res[(var, lv)][0]:14.1f}%"
                  + (f"{v:10.1f}%" if v is not None else "        --"))

    fa = res[("foliage", 0.1)][0]
    ta = [res[("torso", l)][0] for l in LEVELS]
    tm = [minp.get(("torso", l)) for l in LEVELS]
    if all(v is not None for v in tm) and minp.get(("foliage", 0.1)):
        t_at = float(np.interp(fa, ta, tm))
        fm = minp[("foliage", 0.1)]
        print(f"\nfoliage@0.1 destroys {100 * fa:.1f}% of the crop and keeps "
              f"{fm:.1f}% mINP")
        print(f"a torso band destroying the SAME {100 * fa:.1f}% keeps "
              f"{t_at:.1f}% mINP (interpolated)")
        print(f"--> at MATCHED destroyed area, fragmenting the occluder costs "
              f"{t_at / fm:.1f}x more mINP")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["variant", "level", "measured_area", "patches_hit",
                "largest_clean_blob", "n_clean_blobs", "minp_retained_pct"]
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
