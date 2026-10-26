#!/usr/bin/env python3
"""MAINTAINER tool: turn the private evaluation manifests into dataset-free
release manifests (``manifests/``).

What is kept: dataset-relative image ids (file names as they appear in each
dataset's own annotation file), identity labels, query order, a SHA-1 of every
query caption (so a user can prove their local annotation yields the identical
query set without us redistributing the text), and a SHA-256 of every rendered
occluded gallery image (so a user can prove their regenerated occluded gallery
is byte-identical to ours).
What is dropped: caption text, absolute paths, pixels.
"""
import argparse, gzip, hashlib, json, sys
from pathlib import Path

GEOMS = ("torso", "cutout", "foliage")
LEVELS = (10, 20, 30, 40, 50)


def sha1(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def clean_manifest(es):
    return {"name": es["name"], "dataset": es["meta"].get("dataset"), "split": es["meta"].get("split"),
            "split_source": es["meta"].get("split_source"), "stats": es["stats"],
            "gallery": [[g["image_id"], g["pid"]] for g in es["gallery"]],
            "queries": [[q["image_id"], q["pid"], sha1(q["text"])] for q in es["queries"]],
            "columns": {"gallery": ["image_id", "pid"], "queries": ["image_id", "pid", "sha1(caption)"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=["cuhk", "rstp", "icfg"])
    ap.add_argument("--hash-cells", nargs="*", default=None,
                    help="cells to hash, e.g. torso_10 (default: all 15); pass none to skip")
    ap.add_argument("--clean-only", action="store_true")
    a = ap.parse_args()
    R, out = Path(a.result_root), Path(a.out)
    for b in a.benchmarks:
        es = json.loads((R / "evalsets" / f"{b}.json").read_text(encoding="utf-8"))
        dump(clean_manifest(es), out / "evalsets" / f"{b}.json.gz")
        print(b, "clean:", es["stats"])
        if a.clean_only:
            continue
        cells = a.hash_cells if a.hash_cells is not None else [f"{g}_{l}" for g in GEOMS for l in LEVELS]
        for c in cells:
            g, l = c.split("_")
            man = json.loads((R / "occluded" / "manifests" / f"{b}_{g}_l{int(l):02d}.json").read_text(encoding="utf-8"))
            assert [x["image_id"] for x in man["gallery"]] == [x["image_id"] for x in es["gallery"]]
            assert [(q["pid"], q["text"]) for q in man["queries"]] == [(q["pid"], q["text"]) for q in es["queries"]]
            img_dir = R / "occluded" / "images" / f"{b}_{g}_l{int(l):02d}"
            from concurrent.futures import ThreadPoolExecutor
            paths = [img_dir / Path(x["image"]).name for x in man["gallery"]]
            with ThreadPoolExecutor(16) as ex:
                hashes = list(ex.map(sha256_file, paths))
            obj = {"benchmark": b, "geometry": g, "nominal_coverage": int(l) / 100, "select": man["meta"].get("select"),
                   "n_occluded": man["meta"].get("n_occluded"), "jpeg_quality": 95,
                   "file_name_pattern": "{gallery_index:07d}_{stem(image_id)}.jpg",
                   "sha256": hashes}
            dump(obj, out / "occluded" / f"{b}_{g}_l{int(l):02d}.json.gz")
            print(f"  {b} {c}: {len(hashes)} hashes")


if __name__ == "__main__" and "--oof" not in __import__("sys").argv:
    main()


def export_oof(R: Path, out: Path):
    """Out-of-family occluders (real object cut-outs, MGCC's OGor). Only the
    selected gallery images are occluded; the rest keep their clean path."""
    from concurrent.futures import ThreadPoolExecutor
    for b, tag in (("cuhk", "cuhk_ogor_l00_one_per_id"), ("rstp", "rstp_ogor_l50"),
                   ("rstp", "rstp_ogor_l25"), ("rstp", "rstp_ogor_l70"), ("rstp", "rstp_ogor_l00_one_per_id")):
        es = json.loads((R / "evalsets" / f"{b}.json").read_text(encoding="utf-8"))
        man = json.loads((R / "occ_study" / "manifests" / f"{tag}.json").read_text(encoding="utf-8"))
        assert [x["image_id"] for x in man["gallery"]] == [x["image_id"] for x in es["gallery"]]
        occ = [i for i, x in enumerate(man["gallery"]) if "/occ_study/images/" in x["image"].replace("\\\\", "/")]
        paths = [R / "occ_study" / "images" / tag / Path(man["gallery"][i]["image"]).name for i in occ]
        with ThreadPoolExecutor(16) as ex:
            hashes = list(ex.map(sha256_file, paths))
        meta = man["meta"]
        dump({"benchmark": b, "variant": "ogor", "tag": tag, "select": meta.get("select"),
              "level": meta.get("level"), "ogor_native_scale": meta.get("ogor_native_scale"),
              "jitter": meta.get("jitter"), "n_occluded": len(occ), "n_gallery": len(man["gallery"]),
              "occluded_gallery_indices": occ, "sha256": hashes},
             out / "occluded_oof" / f"{tag}.json.gz")
        print(tag, "occluded", len(occ), "of", len(man["gallery"]), meta.get("select"), meta.get("ogor_native_scale"))


if __name__ == "__main__" and "--oof" in sys.argv:
    export_oof(Path(sys.argv[sys.argv.index("--result-root") + 1]), Path(sys.argv[sys.argv.index("--out") + 1]))
