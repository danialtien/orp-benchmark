#!/usr/bin/env python3
"""Fetch the Occlusion Instance Library (OIL) used by MGCC's OGor.

MGCC ships the 60 cut-out PNGs it used -- 15 classes x 4 samples -- inside its
own repo at ``dataset/occlusion_img/``. Downloading *those* files rather than
cutting fresh instances out of COCO is what makes our Occluded-* galleries the
same galleries as the ones behind the published numbers; a visually similar
library of different umbrellas would not be.

    python scripts/fetch_oil.py                       # -> ../Dataset/OIL
    python scripts/fetch_oil.py --out /data/OIL --force

Offline / firewalled machines: clone the repo elsewhere and point at it, or copy
the folder over by hand -- it is ~700 kB total.

    python scripts/fetch_oil.py --from-local /path/to/MGCC/dataset/occlusion_img

Verification is by count and by transparency, not by checksum: the upstream repo
is not versioned by release, so pinning hashes would break on any re-commit
while telling us nothing about whether the images are usable.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sar_person.ogor import (OCCLUSION_CLASSES, SAMPLES_PER_CLASS,  # noqa: E402
                             AREA_RATIO, ANCHOR, _has_alpha)

RAW = ("https://raw.githubusercontent.com/littlexinyi/MGCC/master/"
       "dataset/occlusion_img/{name}")
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "Dataset" / "OIL"


def names() -> list[str]:
    return [f"{c}{i}.png" for c in OCCLUSION_CLASSES
            for i in range(1, SAMPLES_PER_CLASS + 1)]


def ssl_context():
    """Prefer certifi's CA bundle over OpenSSL's compiled-in default paths.

    Plenty of lab servers have no usable system trust store -- ca-certificates
    unpacked but never `update-ca-certificates`d, or a conda OpenSSL looking at
    a prefix that does not exist. Python then fails every HTTPS request with
    CERTIFICATE_VERIFY_FAILED even though the network is fine. certifi ships the
    same Mozilla bundle pip already relies on, so if it is importable, use it.

    Deliberately *not* falling back to unverified TLS: this downloads files that
    end up in every table of the paper, and silently accepting any certificate
    to get them is not a trade worth making. Copy the folder over ssh instead --
    it is 700 kB.
    """
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_one(name: str, dst: Path, timeout: int = 30, ctx=None) -> bool:
    req = urllib.request.Request(RAW.format(name=name),
                                 headers={"User-Agent": "sar_person/fetch_oil"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            data = r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  !! {name}: {e}")
        return False
    if len(data) < 512 or data[:8] != b"\x89PNG\r\n\x1a\n":
        print(f"  !! {name}: not a PNG ({len(data)} bytes)")
        return False
    dst.write_bytes(data)
    return True


def whiten_to_alpha(p: Path, thresh: int = 220) -> None:
    """MGCC's ``transPNG``: make near-white pixels transparent.

    Only for instances that arrived opaque. Pasting an opaque PNG drops a solid
    rectangle -- that is the blackout variant we already have, wearing a car's
    colours -- so an opaque instance is a silent corruption, not a cosmetic one.
    """
    from PIL import Image
    img = Image.open(p).convert("RGBA")
    img.putdata([(255, 255, 255, 0) if (r > thresh and g > thresh and b > thresh)
                 else (r, g, b, a) for r, g, b, a in img.getdata()])
    img.save(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--from-local", default=None,
                    help="copy from an existing MGCC/dataset/occlusion_img")
    ap.add_argument("--force", action="store_true", help="re-download existing")
    ap.add_argument("--whiten-to-alpha", action="store_true",
                    help="apply MGCC's transPNG to any opaque instance")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    wanted = names()

    ctx = None if args.from_local else ssl_context()
    got, failed = [], []
    for name in wanted:
        dst = out / name
        if dst.is_file() and not args.force:
            got.append(name)
            continue
        if args.from_local:
            src = Path(args.from_local).expanduser() / name
            if src.is_file():
                shutil.copy2(src, dst)
                got.append(name)
            else:
                failed.append(name)
            continue
        (got if fetch_one(name, dst, ctx=ctx) else failed).append(name)

    if args.whiten_to_alpha:
        for name in got:
            whiten_to_alpha(out / name)

    # --- verify -----------------------------------------------------------
    from PIL import Image
    opaque = []
    for name in got:
        try:
            if not _has_alpha(Image.open(out / name).convert("RGBA")):
                opaque.append(name)
        except Exception as e:                                # noqa: BLE001
            failed.append(f"{name} (unreadable: {e})")

    manifest = {
        "source": "github.com/littlexinyi/MGCC :: dataset/occlusion_img",
        "n_expected": len(wanted), "n_present": len(got),
        "classes": OCCLUSION_CLASSES, "samples_per_class": SAMPLES_PER_CLASS,
        "area_ratio": AREA_RATIO, "anchor": ANCHOR,
        "missing": sorted(failed), "opaque": sorted(opaque),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")

    print(f"\nOIL -> {out}")
    print(f"  {len(got)}/{len(wanted)} instance(s)")
    if opaque:
        print(f"  {len(opaque)} opaque (no cut-out): {', '.join(opaque[:6])}"
              f"{' ...' if len(opaque) > 6 else ''}\n"
              f"  -> re-run with --whiten-to-alpha, or these become blackout "
              f"rectangles")
    if failed:
        print(f"  MISSING {len(failed)}: {', '.join(sorted(failed)[:8])}")
        print("  An incomplete library changes the occlusion distribution; "
              "occlude_bench will refuse it unless you pass --oil-not-strict.")
        if len(failed) == len(wanted):
            print("\n  Nothing downloaded at all -- this is a network or TLS "
                  "problem, not a missing-file problem. On a machine whose CA "
                  "store is broken (CERTIFICATE_VERIFY_FAILED), do not fight "
                  "it: copy the folder from a machine that already has it.\n"
                  f"    scp -r <local>/Dataset/OIL user@host:{out.parent}/\n"
                  f"    # or:  python scripts/fetch_oil.py --from-local "
                  f"<clone>/MGCC/dataset/occlusion_img\n"
                  "  `pip install certifi` also fixes it in most cases.")
        return 1
    print("  complete -- ready for: occlude_bench --variants ogor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
