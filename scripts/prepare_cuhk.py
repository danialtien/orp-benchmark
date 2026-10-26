"""Turn the author-supplied CUHK-PEDES release into the layout SAR_Person reads.

What the author ships
---------------------
    Dataset/CUHK-PEDES/CUHK-PEDES/caption_all.json   40 206 records
    Dataset/CUHK-PEDES/CUHK-PEDES/imgs/<dir>/<file>  40 206 images, 13 003 ids
    Dataset/CUHK-PEDES/CUHK-PEDES/readme.txt

`caption_all.json` carries ``{id, file_path, captions}`` and **no `split` field**.
Every published CUHK-PEDES number (IRRA, RDE, APTM, CFine ...) is measured on the
split that lives in ``reid_raw.json``, which the 2017 release does not include.
Worse, the identity numbering in the two files is a different permutation: of the
34 054 official train images only 2 happen to carry the same integer id in both.
So the split cannot be re-derived by an id range.

What this script does about it
------------------------------
1. **train — exact, not reconstructed.** ``Dataset/CUHK-PEDES-train`` (built in an
   earlier session straight from ``reid_raw.json``, sha256 in its
   ``provenance.json``) holds the official train file list *and* the official
   reid ids. Every one of its 34 054 paths exists in the author's
   ``caption_all.json``, and the ca-id -> reid-id map it induces is a clean
   bijection over 11 003 identities. Train is therefore recovered bit-exact; only
   the *pixels* change, from unofficial HF mirrors to the author's own files.

2. **val / test — reconstructed, and labelled as such.** The remaining 2 000
   identities (6 152 images) are the union of the official val and test splits,
   but nothing in the author's release says which is which. They are partitioned
   1 000 / 1 000 by a seeded shuffle so the benchmark has the official *shape*,
   and ``cuhk_provenance.json`` records that the partition is ours, not the
   authors'. A second annotation file keeps all 2 000 ids as one test set for
   anyone who would rather not split at all.

Outputs (under ``Dataset/CUHK-PEDES/``)
---------------------------------------
    imgs/                            author images, moved up one level
    caption_all.json                 author annotation, moved up one level
    reid_raw.json                    train + val(1000 ids) + test(1000 ids)
    reid_raw_heldout2000.json        train + test(2000 ids)  [--cuhk-json ...]
    annotations/train.jsonl          training preset input (per-image records)
    cuhk_provenance.json             what is official, what is ours
    annotations/build_report.json    counts, for eyeballing after a copy

Usage
-----
    python scripts/prepare_cuhk.py                       # build (idempotent)
    python scripts/prepare_cuhk.py --verify              # existence + decode
    python scripts/prepare_cuhk.py --verify --fast       # existence only
    python scripts/prepare_cuhk.py --data-root /data/Dataset --verify --fast
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger("prepare_cuhk")

OFFICIAL = {           # Li et al., CVPR 2017 + the reid_raw.json split
    "images": 40206, "ids": 13003, "captions": 80412,
    "train": {"images": 34054, "ids": 11003, "captions": 68126},
    "val": {"images": 3078, "ids": 1000},
    "test": {"images": 3074, "ids": 1000, "captions": 6156},
}


#: How the two possible ``reid_raw.json`` files tell themselves apart.
#: The authors' file carries ``processed_tokens`` and nothing else identifying;
#: :func:`build_records` stamps every reconstructed record with ``author_id``.
def split_kind(path: Path) -> str:
    """``"official"`` | ``"reconstructed"`` | ``"absent"`` | ``"unknown"``.

    The whole point of this script was to survive WITHOUT the authors' split
    file. Once that file is on disk the script must never run over it again --
    a rebuild would silently swap a publishable benchmark for a seeded guess
    that looks identical in every log line. Hence this check, and hence it
    looks at the CONTENT, not at the filename or the mtime.
    """
    if not path.is_file():
        return "absent"
    try:
        recs = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "unknown"
    if not isinstance(recs, list) or not recs:
        return "unknown"
    if "author_id" in recs[0]:
        return "reconstructed"
    counts = {k: 0 for k in ("train", "val", "test")}
    for r in recs:
        if r.get("split") in counts:
            counts[r["split"]] += 1
    if (counts["train"] == OFFICIAL["train"]["images"]
            and counts["val"] == OFFICIAL["val"]["images"]
            and counts["test"] == OFFICIAL["test"]["images"]):
        return "official"
    return "unknown"


# --------------------------------------------------------------------------- #
# layout                                                                       #
# --------------------------------------------------------------------------- #
def normalise_layout(cuhk_root: Path) -> Path:
    """Move the author payload from ``CUHK-PEDES/CUHK-PEDES/`` up one level.

    SAR_Person's defaults (``eval_sets._load_cuhk``) look for
    ``Dataset/CUHK-PEDES/reid_raw.json`` next to ``Dataset/CUHK-PEDES/imgs``.
    The author's zip nests one extra directory; unnesting once here means no
    ``--cuhk-images`` override has to be threaded through every later command.
    """
    inner = cuhk_root / "CUHK-PEDES"
    moved = []
    for name in ("imgs", "caption_all.json", "readme.txt"):
        src, dst = inner / name, cuhk_root / name
        if dst.exists():
            continue
        if src.exists():
            shutil.move(str(src), str(dst))
            moved.append(name)
    if moved:
        log.info("[layout] moved %s out of %s", ", ".join(moved), inner)
    if not (cuhk_root / "imgs").is_dir():
        raise SystemExit(
            f"no imgs/ under {cuhk_root} (nor {inner}). Extract the author's "
            f"release there first.")
    if not (cuhk_root / "caption_all.json").is_file():
        raise SystemExit(f"no caption_all.json under {cuhk_root}")
    return cuhk_root


# --------------------------------------------------------------------------- #
# split recovery                                                               #
# --------------------------------------------------------------------------- #
def load_official_train(train_jsonl: Path):
    """``{author file_path -> official reid person_id}`` for the train split."""
    if not train_jsonl.is_file():
        raise SystemExit(
            f"{train_jsonl} not found.\n"
            f"That file is the only surviving copy of the official train split "
            f"(it was generated from reid_raw.json, sha256 recorded in its "
            f"provenance.json). Without it the train/val/test boundary cannot "
            f"be recovered from the author's caption_all.json alone.\n"
            f"Pass --train-jsonl if it lives somewhere else, or drop "
            f"reid_raw.json into the dataset root and skip this script.")
    out = {}
    with train_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["original_file_path"]] = int(r["person_id"])
    return out


def build_records(author: list, train_pid: dict, seed: int):
    """Assign every author record a split and an official-style integer id."""
    by_path = {r["file_path"]: r for r in author}
    if len(by_path) != len(author):
        raise SystemExit("caption_all.json has duplicate file_path entries")

    missing = [p for p in train_pid if p not in by_path]
    if missing:
        raise SystemExit(
            f"{len(missing)} official train path(s) are absent from the "
            f"author's release, e.g. {missing[:3]} -- the release looks "
            f"incomplete, stop here rather than train on a partial split.")

    # ca-id <-> reid-id must be a bijection over the train identities, otherwise
    # the two files disagree about who is who and nothing below is trustworthy.
    ca2reid, reid2ca = {}, {}
    for path, pid in train_pid.items():
        ca = by_path[path]["id"]
        if ca2reid.setdefault(ca, pid) != pid or reid2ca.setdefault(pid, ca) != ca:
            raise SystemExit(
                f"identity mapping is not 1:1 at {path} "
                f"(author id {ca} <-> official id {pid})")

    heldout_ca = sorted(set(r["id"] for r in author) - set(ca2reid))
    rng = random.Random(seed)
    shuffled = list(heldout_ca)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    val_ca, test_ca = sorted(shuffled[:half]), sorted(shuffled[half:])

    # keep the official id ranges: train 1..11003, val 11004.., test ..13003
    next_id = max(ca2reid.values()) + 1
    for ca in val_ca:
        ca2reid[ca] = next_id
        next_id += 1
    for ca in test_ca:
        ca2reid[ca] = next_id
        next_id += 1

    split_of = {ca: "val" for ca in val_ca}
    split_of.update({ca: "test" for ca in test_ca})

    records = []
    for r in author:                       # author order, deterministic
        ca = r["id"]
        path = r["file_path"]
        split = "train" if path in train_pid else split_of[ca]
        caps = [c.strip() for c in r["captions"] if c and c.strip()]
        records.append({
            "split": split,
            "captions": caps,
            "file_path": path,
            "id": ca2reid[ca],
            "author_id": ca,
            "image_id": path,          # per-image key; `id` is the identity
        })
    records.sort(key=lambda r: (["train", "val", "test"].index(r["split"]), r["id"],
                                r["file_path"]))
    return records, {"val_ids": len(val_ca), "test_ids": len(test_ca)}


# --------------------------------------------------------------------------- #
# writers                                                                      #
# --------------------------------------------------------------------------- #
def _stats(records, split=None):
    rs = [r for r in records if split is None or r["split"] == split]
    return {"images": len(rs), "ids": len(set(r["id"] for r in rs)),
            "captions": sum(len(r["captions"]) for r in rs)}


def write_train_files(ann_dir: Path, train: list, source: str):
    """Write ``train.jsonl`` + ``train_captions.csv`` from train records.

    ``train`` is a list of reid-style dicts (``file_path``/``id``/``captions``).
    The captions are taken verbatim, which is the whole point: reid_raw.json
    carries the CLEANED text (apostrophes, "t-shirt", "90 degrees"), while
    caption_all.json carries a damaged copy of the same sentences. Training on
    the reid_raw text is what makes "same data as IRRA" literally true.
    """
    ann_dir.mkdir(parents=True, exist_ok=True)
    with (ann_dir / "train.jsonl").open("w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps({
                "image_id": r["file_path"],
                "file_name": f"imgs/{r['file_path']}",
                "original_file_path": r["file_path"],
                "person_id": r["id"],
                "split": "train",
                "captions": [c.strip() for c in r["captions"] if c and c.strip()],
                "source": source,
            }, ensure_ascii=False) + "\n")
    with (ann_dir / "train_captions.csv").open("w", encoding="utf-8",
                                               newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "file_name", "person_id", "caption"])
        for r in train:
            for c in r["captions"]:
                if c and c.strip():
                    w.writerow([r["file_path"], f"imgs/{r['file_path']}",
                                r["id"], c.strip()])


def rebuild_train_from_official(cuhk_root: Path) -> int:
    """Point the training preset at the authors' own caption text."""
    recs = json.loads((cuhk_root / "reid_raw.json").read_text(encoding="utf-8"))
    train = sorted((r for r in recs if r.get("split") == "train"),
                   key=lambda r: (r["id"], r["file_path"]))
    write_train_files(cuhk_root / "annotations", train, "official-reid_raw")
    log.info("[train ] rebuilt from reid_raw.json: %d image(s), %d id(s), "
             "%d caption(s)", len(train), len({r["id"] for r in train}),
             sum(len(r["captions"]) for r in train))
    return len(train)


def write_annotations(cuhk_root: Path, records: list, seed: int,
                      reid_name: str = "reid_raw.json",
                      annotations: bool = True):
    ann_dir = cuhk_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # 1. the reconstruction -- normally reid_raw.json, the path
    #    eval_sets._load_cuhk reads by default. When the authors' own file is
    #    already sitting there, main() redirects us to reid_raw_generated.json
    #    instead so the publishable split is never overwritten.
    (cuhk_root / reid_name).write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8")

    # 2. the "don't split what you can't split" variant: one 2 000-id test set
    heldout = [dict(r, split="test" if r["split"] != "train" else "train")
               for r in records]
    (cuhk_root / "reid_raw_heldout2000.json").write_text(
        json.dumps(heldout, ensure_ascii=False), encoding="utf-8")

    # 3. per-image train records for the training preset. `file_name` is
    #    relative to Dataset/CUHK-PEDES so image_root can stay a directory that
    #    also holds the annotations.
    train = [r for r in records if r["split"] == "train"]
    if annotations:
        write_train_files(ann_dir, train, "author-release")

    report = {
        "total": _stats(records),
        "train": _stats(records, "train"),
        "val": _stats(records, "val"),
        "test": _stats(records, "test"),
        "heldout_all": {"images": _stats(records, "val")["images"]
                        + _stats(records, "test")["images"],
                        "ids": _stats(records, "val")["ids"]
                        + _stats(records, "test")["ids"]},
        "official": OFFICIAL,
        "split_seed": seed,
    }
    if annotations:
        (ann_dir / "build_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_provenance(cuhk_root: Path, report: dict, seed: int, train_jsonl: Path):
    (cuhk_root / "cuhk_provenance.json").write_text(json.dumps({
        "dataset": "CUHK-PEDES",
        "images": {
            "source": "author release (Li et al., CVPR 2017), obtained on "
                      "request with a university email",
            "status": "OFFICIAL -- publishable",
            "path": "Dataset/CUHK-PEDES/imgs",
        },
        "captions": {
            "source": "caption_all.json from the same author release",
            "status": "OFFICIAL",
        },
        "splits": {
            "train": {
                "status": "OFFICIAL",
                "how": "file list + identity numbering taken from "
                       f"{train_jsonl.as_posix()}, which was generated from "
                       "reid_raw.json (sha256 fc3d15542f91dd70e23028c2cb4017"
                       "1a8f41734a39afddb9505e352e396cabf0). All 34 054 paths "
                       "verified present in the author release; the author-id "
                       "to reid-id map is a verified bijection over 11 003 ids.",
            },
            "val/test": {
                "status": "RECONSTRUCTED -- not the authors' partition",
                "how": "caption_all.json carries no `split` field, so the 2 000 "
                       "non-train identities were split 1 000/1 000 by a "
                       f"shuffle with seed={seed}. Sizes match the published "
                       "protocol; the identity membership does not.",
                "implication": "R@K / mAP on this test split are NOT strictly "
                               "comparable to IRRA / RDE / APTM tables. Report "
                               "them as 'CUHK-PEDES (reconstructed test split)' "
                               "or obtain reid_raw.json and rebuild.",
                "alternative": "reid_raw_heldout2000.json keeps all 2 000 "
                               "held-out identities as a single test set "
                               "(harder, larger gallery, nothing invented).",
            },
        },
        "counts": report,
        "terms_of_use": "Non-commercial research only. Do not redistribute the "
                        "images (readme.txt, CUHK terms).",
    }, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# verification                                                                 #
# --------------------------------------------------------------------------- #
def verify(cuhk_root: Path, records: list, fast: bool, workers: int) -> int:
    """Every referenced image exists (and, unless --fast, actually decodes).

    Run this again after copying to the training server: a truncated scp
    produces files that pass an existence check and explode mid-epoch.
    """
    imgs = cuhk_root / "imgs"
    Image = None
    if not fast:
        try:
            from PIL import Image  # noqa: F811
        except ImportError:
            log.warning("[verify] Pillow missing -- existence check only")
            fast = True

    def check(rec):
        p = imgs / rec["file_path"]
        if not p.is_file():
            return ("missing", rec["file_path"])
        if fast:
            return None
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as exc:                       # noqa: BLE001
            return (f"corrupt:{type(exc).__name__}", rec["file_path"])
        return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        bad = [b for b in ex.map(check, records) if b]

    if bad:
        log.error("[verify] %d bad file(s); first 10:", len(bad))
        for why, p in bad[:10]:
            log.error("   %-20s %s", why, p)
    else:
        log.info("[verify] %d image(s) OK (%s)", len(records),
                 "existence" if fast else "existence + decode")
    return len(bad)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    here = Path(__file__).resolve().parents[2]
    ap.add_argument("--data-root", default=str(here / "Dataset"))
    ap.add_argument("--cuhk-dir", default=None,
                    help="default: <data-root>/CUHK-PEDES")
    ap.add_argument("--train-jsonl", default=None,
                    help="default: <data-root>/CUHK-PEDES-train/annotations/"
                         "train.jsonl (the official train file list)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the val/test reconstruction; changing it "
                         "changes the benchmark, so never change it silently")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="with --verify: existence only, skip image decode")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-build", action="store_true",
                    help="only verify against the annotations already on disk")
    ap.add_argument("--rebuild-train", action="store_true",
                    help="rebuild annotations/train.jsonl + train_captions.csv "
                         "from the reid_raw.json on disk (the authors' caption "
                         "text) and stop. Use after dropping in the official "
                         "split file.")
    ap.add_argument("--rebuild-reconstruction", action="store_true",
                    help="build the seeded val/test reconstruction even though "
                         "the authors' reid_raw.json is present. It is written "
                         "to reid_raw_generated.json; the official file is "
                         "never overwritten.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(args.data_root).resolve()
    cuhk = Path(args.cuhk_dir) if args.cuhk_dir else root / "CUHK-PEDES"
    train_jsonl = Path(args.train_jsonl) if args.train_jsonl else (
        root / "CUHK-PEDES-train" / "annotations" / "train.jsonl")

    normalise_layout(cuhk)

    kind = split_kind(cuhk / "reid_raw.json")
    log.info("[split] reid_raw.json on disk: %s", kind.upper())

    # The authors' split file outranks anything this script can produce. Refuse
    # the rebuild rather than quietly replacing 3078/3074 with 3067/3085.
    if kind == "official" and not (args.no_build or args.rebuild_train
                                   or args.rebuild_reconstruction):
        raise SystemExit(
            "reid_raw.json in %s is the AUTHORS' official split "
            "(train/val/test = %d/%d/%d) -- rebuilding would overwrite it with "
            "the seeded reconstruction.\n"
            "  * verify what is on disk:            --no-build --verify\n"
            "  * rebuild train.jsonl from it:       --rebuild-train\n"
            "  * rebuild the seeded reconstruction: "
            "--rebuild-reconstruction (writes reid_raw_generated.json, "
            "leaves reid_raw.json and annotations/ alone)"
            % (cuhk, OFFICIAL["train"]["images"], OFFICIAL["val"]["images"],
               OFFICIAL["test"]["images"]))

    if args.rebuild_train:
        if kind != "official":
            raise SystemExit(
                "--rebuild-train reads the AUTHORS' reid_raw.json, but the "
                f"file on disk is {kind.upper()}. Put the official split file "
                "at %s first." % (cuhk / "reid_raw.json"))
        rebuild_train_from_official(cuhk)
        return

    if args.no_build:
        records = json.loads((cuhk / "reid_raw.json").read_text(encoding="utf-8"))
        # build_report.json describes whichever file was built LAST, which is
        # not necessarily the one now at reid_raw.json -- recompute instead.
        report = {"total": _stats(records),
                  "train": _stats(records, "train"),
                  "val": _stats(records, "val"),
                  "test": _stats(records, "test"),
                  "official": OFFICIAL, "split_seed": args.seed}
    else:
        author = json.loads((cuhk / "caption_all.json").read_text(encoding="utf-8"))
        log.info("[author] %d record(s), %d identit(ies), %d caption(s)",
                 len(author), len(set(r["id"] for r in author)),
                 sum(len(r["captions"]) for r in author))
        records, _ = build_records(author, load_official_train(train_jsonl),
                                   args.seed)
        official_present = kind == "official"
        reid_name = ("reid_raw_generated.json" if official_present
                     else "reid_raw.json")
        # With the authors' file on disk, train.jsonl / train_captions.csv /
        # build_report.json describe THAT split, not this reconstruction --
        # rewriting them here would quietly swap the training captions back to
        # the damaged caption_all.json text.
        report = write_annotations(cuhk, records, args.seed, reid_name,
                                   annotations=not official_present)
        log.info("[write] reconstruction -> %s", cuhk / reid_name)
        if official_present:
            log.info("[write] annotations/ and cuhk_provenance.json left "
                     "as-is: the split in use is the authors', not this "
                     "reconstruction")
        else:
            write_provenance(cuhk, report, args.seed, train_jsonl)

    # `report` describes whatever was just built or just read -- which is NOT
    # always the split that eval will use. Say which file these counts are for.
    report_official = args.no_build and kind == "official"
    log.info("[counts] describing %s", "reid_raw.json (OFFICIAL)"
             if report_official else
             "the seeded reconstruction" if not args.no_build
             else (cuhk / "reid_raw.json").name)
    for k in ("total", "train", "val", "test"):
        exp = OFFICIAL if k == "total" else OFFICIAL.get(k, {})
        got = report[k]
        flag = ""
        for field in ("images", "ids", "captions"):
            if field in exp and field in got and exp[field] != got[field]:
                flag += f"  [{field}: official {exp[field]}]"
        log.info("[%-5s] %6d image(s)  %5d id(s)  %6d caption(s)%s",
                 k, got["images"], got["ids"], got["captions"], flag)
    if report_official:
        log.info("[note ] val/test are the AUTHORS' OFFICIAL split -- these "
                 "numbers are directly comparable to IRRA / RDE / APTM")
    elif kind == "official":
        log.info("[note ] the counts above are the RECONSTRUCTION's "
                 "(seed=%d). The split eval actually uses is the authors' "
                 "reid_raw.json: 34054 / 3078 / 3074", report["split_seed"])
    else:
        log.info("[note ] val/test are RECONSTRUCTED (seed=%d) -- see "
                 "cuhk_provenance.json before quoting these numbers in a paper",
                 report["split_seed"])

    if args.verify and verify(cuhk, records, args.fast, args.workers):
        sys.exit(1)


if __name__ == "__main__":
    main()
