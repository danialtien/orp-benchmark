#!/usr/bin/env python3
"""Turn NOMAD into a CLIP+LoRA training set: person crops + one shared caption.

NOMAD ships 42,825 selected frames at 5472x3078, each with a Detectron2-format
bounding box and a visibility label, plus a SINGLE appearance description per
actor in ``annotations/metadata.json``. Training CLIP on the raw frames is
pointless -- the person occupies 0.03%..0.9% of the pixels -- so this script
crops every qualifying box into ``NOMAD/crop/images/ActorNNN/`` and pairs it with
that actor's caption. Many crops, one description: exactly the supervision that
teaches the text tower which pixels the attribute words refer to.

What it writes (all under ``--out``, default ``NOMAD/crop``)::

    images/ActorNNN/ActorNNN_a10_f0114_p0.jpg   the person ROIs
    annotations.json    flat [{file_path, captions, id, person_id, split, ...}]
    nomad_profiles.json per-actor PersonProfileSchema attributes + all three
                        prompt flavours (DEFAULT / APTM templated / +negatives)
    build_report.json   filter counts, crop-size percentiles, split sizes
    .crop_progress      resume journal: "tag<TAB>w<TAB>h" per finished crop

Design choices worth knowing before you change a flag
-----------------------------------------------------
* **Altitudes.** ``--altitudes a10 a30`` by default. Median person height is
  158 px at 10 m and 59 px at 30 m, but only 37/26/21 px at 50/70/90 m -- below
  that the garment colours the caption asserts are not in the pixels at all, and
  the contrastive loss just learns noise.
* **Visibility.** ``--min-visibility 50``. NOMAD labels the fraction of body
  inside the box (10..100). A 20%-visible actor is a shoe behind a bush; pairing
  it with a head-to-toe description is a mislabelled positive.
* **Actor-level split.** The 80/20 train/val split is over ACTORS, never over
  crops, because every crop of an actor carries the identical caption -- a
  crop-level split would leak val captions into training and inflate R@1.
* **Padding.** ``--pad 0.12`` expands the box before cropping. NOMAD boxes are
  tight; some context helps the ViT patches at the silhouette boundary. Crops are
  saved at native resolution AND native aspect ratio -- ``ClipEncoder._prep_image``
  letterboxes them, so nothing gets truncated downstream.

Resuming
--------
Every finished crop is journalled to ``.crop_progress``, so an interrupted build
picks up where it stopped and costs nothing for work already done. Two ways to
run it in bounded steps (useful under a per-command wall-clock limit):

    --time-budget SEC   decode for SEC seconds, journal, report what is left
    --shard I/N          decode only shard I of N

Either way, run ``--finalize`` once at the end to write annotations.json from the
journal. A plain run with no bounding flag does everything and finalizes itself.

Examples
--------
# Count what would be produced, decode nothing:
python scripts/prepare_nomad.py --dry-run

# The default build (a10+a30, visibility >= 50, 80/20 by actor):
python scripts/prepare_nomad.py --workers 8

# All altitudes, keep everything, filter later at train time:
python scripts/prepare_nomad.py --altitudes a10 a30 a50 a70 a90 \
    --min-visibility 10 --out ../Dataset/NOMAD/crop_all
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sar_person.nomad_meta import (                          # noqa: E402
    actor_dir_name, build_actor_captions, load_nomad_metadata,
)

log = logging.getLogger("prepare_nomad")

ALL_ALTITUDES = ["a10", "a30", "a50", "a70", "a90"]
PROGRESS_NAME = ".crop_progress"


# --------------------------------------------------------------------------- #
# frame-name parsing / geometry                                               #
# --------------------------------------------------------------------------- #

def parse_frame(file_name: str) -> tuple[str, str, str]:
    """'Actor001_a10_f0114.jpg' -> ('Actor001', 'a10', '0114')."""
    parts = Path(file_name).stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"unexpected NOMAD frame name: {file_name!r}")
    return parts[0], parts[1], parts[2].lstrip("f")


def to_xyxy(bbox, mode: int) -> tuple[float, float, float, float]:
    """Detectron2 BoxMode -> absolute (x1, y1, x2, y2).

    NOMAD uses ``bbox_mode: 1`` == XYWH_ABS; mode 0 is XYXY_ABS. Any other mode
    (rotated boxes) does not occur in NOMAD and is rejected loudly rather than
    silently mis-cropped.
    """
    x, y, a, b = (float(v) for v in bbox)
    if mode == 1:
        return x, y, x + a, y + b
    if mode == 0:
        return x, y, a, b
    raise ValueError(f"unsupported bbox_mode {mode}")


# --------------------------------------------------------------------------- #
# activity labels: per-frame metadata, deliberately NOT part of the caption   #
# --------------------------------------------------------------------------- #

def load_activity_index(path: Path) -> dict:
    """``activityLabels.json`` -> {(actor, altitude): [(start, end, label)]}.

    Posture is per frame while the caption must be actor-invariant, so activity
    is recorded per crop instead of written into the text. That lets you slice
    the set later (``--nomad-activities Walking``) without re-cropping.
    """
    idx: dict = {}
    if not path.is_file():
        log.warning("[activity] %s not found; activity fields stay empty", path)
        return idx
    bad = 0
    for rec in json.loads(path.read_text(encoding="utf-8")):
        actor = actor_dir_name(rec["id"])
        for alt_key, classes in (rec.get("labels") or {}).items():
            spans = []
            for label, ranges in classes.items():
                for pair in ranges:
                    if len(pair) != 2:
                        continue
                    # A few span endpoints are typo'd in the shipped file
                    # (e.g. "1320}"). Keep the digits; this is optional
                    # metadata and must never abort a build.
                    a = re.sub(r"\D", "", str(pair[0]))
                    b = re.sub(r"\D", "", str(pair[1]))
                    if a and b:
                        spans.append((int(a), int(b), label))
                    else:
                        bad += 1
            idx[(actor, f"a{alt_key}")] = sorted(spans)
    if bad:
        log.warning("[activity] skipped %d unparsable span(s) in %s", bad,
                    path.name)
    return idx


def activity_for(idx: dict, actor: str, altitude: str, frame: str) -> str:
    try:
        f = int(frame)
    except ValueError:
        return ""
    for start, end, label in idx.get((actor, altitude), ()):
        if start <= f <= end:
            return label
    return ""


# --------------------------------------------------------------------------- #
# worker                                                                      #
# --------------------------------------------------------------------------- #

def crop_one_frame(task: tuple) -> tuple:
    """Decode ONE frame, write its qualifying crops, return their sizes.

    The payload is a minimal tuple -- ``(src, out_dir, pad, min_side, quality,
    [(tag, x1, y1, x2, y2), ...])``. Captions never cross the process boundary:
    an earlier version shipped the full annotation record (three ~500-char
    prompt strings per box) to every worker and pickling dominated the wall
    clock. The parent owns the metadata; it only needs crop dimensions back.

    Boxes are grouped by frame so each 17 MP JPEG is opened exactly once even
    when it holds several people.
    """
    from PIL import Image

    src, out_dir_s, pad, min_side, quality, boxes = task
    out_dir = Path(out_dir_s)
    sizes: dict = {}

    pending = []
    for tag, x1, y1, x2, y2 in boxes:
        dst = out_dir / f"{tag}.jpg"
        # Resume must verify, not just stat: a crop left truncated by an
        # interrupted earlier run still "exists", and Image.open() reads the
        # header lazily so it looks healthy here too. Checking the EOI marker
        # costs 2 bytes and means a damaged crop is always regenerated (the
        # atomic save below overwrites it, so a failed unlink doesn't matter).
        if dst.exists() and jpeg_intact(dst):
            try:
                with Image.open(dst) as im:
                    sizes[tag] = im.size
                continue
            except OSError:
                pass
        pending.append((tag, x1, y1, x2, y2))
    if not pending:
        return "reused", None, sizes

    try:
        im = Image.open(src)
        im.load()
        im = im.convert("RGB")
    except (FileNotFoundError, OSError) as exc:
        return "error", f"{Path(src).name}: {exc}", sizes

    W, H = im.size
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, x1, y1, x2, y2 in pending:
        w, h = x2 - x1, y2 - y1
        px, py = w * pad, h * pad          # symmetric padding, clamped to frame
        cx1 = max(0, int(round(x1 - px)));  cy1 = max(0, int(round(y1 - py)))
        cx2 = min(W, int(round(x2 + px)));  cy2 = min(H, int(round(y2 + py)))
        cw, ch = cx2 - cx1, cy2 - cy1
        if min(cw, ch) < min_side:
            sizes[tag] = "skip"
            continue
        # Atomic write: encode to a .tmp sibling, then rename. Saving straight to
        # the final path means a process killed mid-encode (SIGKILL, OOM, a
        # wall-clock limit) leaves a TRUNCATED .jpg that looks fine to
        # Image.open() -- PIL reads the header lazily -- and only explodes 20
        # epochs into training. os.replace is atomic within a filesystem, so an
        # interrupted run can leave an orphan .tmp but never a corrupt crop.
        dst = out_dir / f"{tag}.jpg"
        tmp = out_dir / f"{tag}.jpg.tmp"
        im.crop((cx1, cy1, cx2, cy2)).save(tmp, "JPEG", quality=quality)
        os.replace(tmp, dst)
        sizes[tag] = (cw, ch)
    return "ok", None, sizes


def jpeg_intact(path) -> bool:
    """True if `path` is a complete JPEG (ends with the EOI marker FF D9).

    Two bytes instead of a full decode: this catches exactly the truncated-write
    failure mode, ~1000x cheaper, so verifying the whole set stays practical.
    """
    try:
        if os.path.getsize(path) < 4:
            return False
        with open(path, "rb") as fh:
            fh.seek(-2, 2)
            return fh.read(2) == b"\xff\xd9"
    except OSError:
        return False


def to_task(job: dict) -> tuple:
    """Planned job -> the tuple ``crop_one_frame`` needs."""
    return (job["src"], job["out_dir"], job["pad"], job["min_side"],
            job["quality"], [(b["tag"], *b["xyxy"]) for b in job["boxes"]])


# --------------------------------------------------------------------------- #
# planning                                                                    #
# --------------------------------------------------------------------------- #

def assign_splits(actor_ids, val_frac: float, seed: int) -> dict:
    """Deterministic ACTOR-level train/val assignment (no identity leakage)."""
    import random
    ids = sorted(actor_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    val = set(ids[:n_val])
    return {a: ("val" if a in val else "train") for a in sorted(ids)}


def build_plan(args) -> tuple[list, dict, dict, dict]:
    """Scan the NOMAD annotations and produce (jobs, counters, captions, splits)."""
    nomad_root = Path(args.nomad_root)
    meta_by_actor = load_nomad_metadata(
        nomad_root / "annotations" / "metadata.json")
    if args.limit_actors:
        keep = sorted(meta_by_actor)[:args.limit_actors]
        meta_by_actor = {k: meta_by_actor[k] for k in keep}
    log.info("[meta] %d actor(s) described in metadata.json", len(meta_by_actor))

    splits = assign_splits(list(meta_by_actor), args.val_frac, args.seed)
    captions = {}
    for actor, rec in meta_by_actor.items():
        c = build_actor_captions(rec, skin_tone=args.skin_tone,
                                 upper_layer=args.upper_layer)
        c["split"] = splits[actor]
        c["environment"] = (rec.get("environmental") or {}).get("location", "")
        c["actor_id"] = rec.get("id")
        captions[actor] = c
    n_val = sum(1 for v in splits.values() if v == "val")
    log.info("[split] %d train actor(s) / %d val actor(s) (seed=%d)",
             len(splits) - n_val, n_val, args.seed)

    activity_idx = load_activity_index(
        nomad_root / "annotations" / "activityLabels.json")

    records = json.loads(
        (nomad_root / "annotations" / "annotations.json").read_text(
            encoding="utf-8"))
    altitudes = set(args.altitudes)
    per_group: dict = defaultdict(int)
    counters: dict = defaultdict(int)
    jobs = []

    for rec in records:
        counters["frames_total"] += 1
        file_name = rec.get("file_name") or rec.get("image_id")
        try:
            actor, altitude, frame = parse_frame(file_name)
        except ValueError:
            counters["skip_badname"] += 1
            continue
        if altitude not in altitudes:
            counters["skip_altitude"] += 1
            continue
        if actor not in meta_by_actor:
            counters["skip_no_metadata"] += 1      # e.g. the TN (true-negative) dir
            continue
        anns = rec.get("annotations") or []
        if not anns:
            counters["skip_no_box"] += 1
            continue
        if args.max_frames_per_group:
            key = (actor, altitude)
            if per_group[key] >= args.max_frames_per_group:
                counters["skip_group_cap"] += 1
                continue
            per_group[key] += 1

        act = activity_for(activity_idx, actor, altitude, frame)
        boxes = []
        for pi, a in enumerate(anns):
            try:
                vis = int(str(a.get("visibility", "0")).strip())
            except ValueError:
                vis = 0
            if vis < args.min_visibility:
                counters["skip_visibility"] += 1
                continue
            try:
                xyxy = to_xyxy(a["bbox"], int(a.get("bbox_mode", 1)))
            except (KeyError, ValueError, TypeError):
                counters["skip_badbox"] += 1
                continue
            if (xyxy[2] - xyxy[0]) < 2 or (xyxy[3] - xyxy[1]) < 2:
                counters["skip_degenerate"] += 1
                continue
            boxes.append({
                "tag": f"{actor}_{altitude}_f{frame}_p{pi}",
                "xyxy": xyxy,
                "person_id": actor,
                "altitude_m": int(altitude[1:]),
                "visibility": vis,
                "activity": act,
                "frame": frame,
                "src_frame": f"{actor}/{actor}_{altitude}/{file_name}",
            })
        if not boxes:
            continue
        counters["frames_kept"] += 1
        counters["boxes_kept"] += len(boxes)
        jobs.append({
            "src": str(nomad_root / "images" / actor / f"{actor}_{altitude}"
                       / file_name),
            "out_dir": str(Path(args.out) / "images" / actor),
            "actor": actor, "boxes": boxes, "pad": args.pad,
            "min_side": args.min_side, "quality": args.jpeg_quality,
        })
    return jobs, dict(counters), captions, splits


# --------------------------------------------------------------------------- #
# resume journal                                                              #
# --------------------------------------------------------------------------- #

def read_progress(path: Path) -> dict:
    """``.crop_progress`` -> {tag: (w, h) | "skip"}."""
    done: dict = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[1] != "skip":
            try:
                done[parts[0]] = (int(parts[1]), int(parts[2]))
            except ValueError:
                pass
        elif len(parts) >= 2 and parts[1] == "skip":
            done[parts[0]] = "skip"
    return done


def verify_crops_standalone(out: Path, workers: int) -> int:
    """Check every crop already on disk, without needing the source annotations.

    The plan-based :func:`verify_crops` enumerates the crops it *expects* from
    ``annotations/annotations.json``, which lives with the 344 GB source images
    and is deliberately never copied to a training server. But verifying after
    an rsync is exactly what a training server needs to do -- an interrupted
    transfer leaves a truncated JPEG that only fails mid-epoch. So when the
    annotations are absent, walk the crop tree instead.

    The trade-off is stated rather than hidden: this finds *corrupt* crops, and
    (via build_report.json) a wrong *count*, but it cannot tell you which crop
    is missing, because without the plan there is no list of what should exist.
    Re-running the rsync is the fix either way.
    """
    from concurrent.futures import ThreadPoolExecutor

    present = sorted(p for p in (out / "images").rglob("*.jpg"))
    if not present:
        raise SystemExit(f"no crops under {out / 'images'} -- nothing to verify")
    with ThreadPoolExecutor(max(8, workers * 4)) as ex:
        ok = list(ex.map(jpeg_intact, present))
    corrupt = [p for p, good in zip(present, ok) if not good]

    for p in corrupt:
        try:
            p.unlink()
        except OSError as exc:
            log.warning("[verify] could not delete %s (%s)", p.name, exc)

    orphans = 0
    for t in (out / "images").rglob("*.jpg.tmp"):
        t.unlink(missing_ok=True)
        orphans += 1

    log.info("[verify] %d crop(s) checked, %d corrupt (deleted), %d orphan "
             ".tmp swept  [standalone: no source annotations]",
             len(present), len(corrupt), orphans)

    report = out / "build_report.json"
    if report.is_file():
        try:
            r = json.loads(report.read_text(encoding="utf-8"))
            expected = r.get("crops_indexed")
        except (json.JSONDecodeError, OSError):
            expected = None
        if expected is not None:
            have = len(present) - len(corrupt)
            if have != expected:
                log.error("[verify] %d intact crop(s) but build_report.json "
                          "says %d -- the transfer is INCOMPLETE. Re-run the "
                          "rsync with --checksum (a truncated file often keeps "
                          "a plausible mtime, so size+time skips it).",
                          have, expected)
                return 1
            log.info("[verify] count matches build_report.json (%d)", expected)
    else:
        log.warning("[verify] no build_report.json next to the crops, so the "
                    "count could not be checked -- copy it too")
    if corrupt:
        for p in corrupt[:10]:
            log.error("[verify]   corrupt: %s", p.name)
        return 1
    return 0


def verify_crops(out: Path, jobs: list, done: dict, workers: int) -> dict:
    """Delete every incomplete crop and forget it, so a re-run regenerates it.

    Returns the pruned journal dict. Also sweeps orphan ``.jpg.tmp`` files left
    by an interrupted atomic write.
    """
    from concurrent.futures import ThreadPoolExecutor

    targets = [(b["tag"], Path(j["out_dir"]) / f"{b['tag']}.jpg")
               for j in jobs for b in j["boxes"]]
    present = [(tag, p) for tag, p in targets if p.exists()]
    # I/O-bound, not CPU-bound: threads, and plenty of them.
    with ThreadPoolExecutor(max(8, workers * 4)) as ex:
        ok = list(ex.map(lambda tp: jpeg_intact(tp[1]), present))
    corrupt = [(tag, p) for (tag, p), good in zip(present, ok) if not good]

    for tag, p in corrupt:
        try:
            p.unlink()
        except OSError as exc:
            # Not fatal: crop_one_frame re-validates on resume and the atomic
            # save overwrites in place, so an undeletable corrupt file is still
            # regenerated. Only the journal entry has to go.
            log.warning("[verify] could not delete %s (%s); it will be "
                        "overwritten on the next run instead", p.name, exc)
        done.pop(tag, None)

    orphans = 0
    for j in {j["out_dir"] for j in jobs}:
        for t in Path(j).glob("*.jpg.tmp"):
            t.unlink(missing_ok=True)
            orphans += 1

    # rewrite the journal without the deleted tags
    prog = out / PROGRESS_NAME
    with open(prog, "w", encoding="utf-8") as fh:
        append_progress(fh, done)

    log.info("[verify] %d crop(s) checked, %d corrupt (deleted), %d orphan "
             ".tmp swept", len(present), len(corrupt), orphans)
    if corrupt:
        for tag, _ in corrupt[:10]:
            log.warning("[verify] regenerate needed: %s", tag)
        log.warning("[verify] re-run WITHOUT --verify to rebuild the %d "
                    "deleted crop(s)", len(corrupt))
    return done


def append_progress(fh, sizes: dict) -> None:
    for tag, v in sizes.items():
        if v == "skip":
            fh.write(f"{tag}\tskip\tskip\n")
        elif v:
            fh.write(f"{tag}\t{v[0]}\t{v[1]}\n")
    fh.flush()


# --------------------------------------------------------------------------- #
# outputs                                                                     #
# --------------------------------------------------------------------------- #

def percentiles(vals) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    def p(q):
        return s[min(len(s) - 1, int(q * len(s)))]
    return {"min": s[0], "p10": p(0.10), "p50": p(0.50), "p90": p(0.90),
            "max": s[-1], "mean": round(sum(s) / len(s), 1)}


def finalize(args, out: Path, jobs, counters, captions, splits, done: dict):
    """Write annotations.json / nomad_profiles.json / build_report.json.

    Built purely from the plan plus the resume journal -- no image is touched, so
    this is a couple of seconds even for the full set.
    """
    from PIL import Image

    prompt_key = f"prompt_{args.prompt_style}"
    annotations, missing, skipped, recovered = [], 0, 0, 0
    for j in jobs:
        for b in j["boxes"]:
            size = done.get(b["tag"])
            if size is None:
                # The journal is advisory, not authoritative: a crop written
                # before journalling existed (or by an older run) is still valid
                # data. Fall back to the file itself before calling it missing.
                dst = Path(j["out_dir"]) / f"{b['tag']}.jpg"
                if dst.exists():
                    try:
                        with Image.open(dst) as im:
                            size = im.size
                        recovered += 1
                    except OSError:
                        missing += 1
                        continue
                else:
                    missing += 1
                    continue
            if size == "skip":
                skipped += 1
                continue
            actor = b["person_id"]
            cap = captions[actor]
            annotations.append({
                "id": b["tag"],
                "file_path": f"{actor}/{b['tag']}.jpg",
                "captions": [cap[prompt_key]],
                "person_id": actor,
                "actor_id": cap.get("actor_id"),
                "split": cap["split"],
                "altitude_m": b["altitude_m"],
                "visibility": b["visibility"],
                "activity": b["activity"],
                "frame": b["frame"],
                "environment": cap.get("environment", ""),
                "crop_w": size[0], "crop_h": size[1],
                "src_frame": b["src_frame"],
                "bbox_xyxy_src": [round(v, 1) for v in b["xyxy"]],
                "prompt_default": cap["prompt_default"],
                "prompt_templated": cap["prompt_templated"],
                "prompt_templated_neg": cap["prompt_templated_neg"],
            })
    annotations.sort(key=lambda r: r["id"])

    per_actor: dict = defaultdict(int)
    per_split: dict = defaultdict(int)
    per_alt: dict = defaultdict(int)
    per_act: dict = defaultdict(int)
    for r in annotations:
        per_actor[r["person_id"]] += 1
        per_split[r["split"]] += 1
        per_alt[r["altitude_m"]] += 1
        per_act[r["activity"] or "unlabelled"] += 1

    (out / "annotations.json").write_text(json.dumps(annotations, indent=1),
                                          encoding="utf-8")
    (out / "nomad_profiles.json").write_text(
        json.dumps(captions, indent=2, ensure_ascii=False), encoding="utf-8")

    n_val = sum(1 for v in splits.values() if v == "val")
    report = {
        "source": str(Path(args.nomad_root)),
        "filters": {"altitudes": args.altitudes,
                    "min_visibility": args.min_visibility,
                    "pad": args.pad, "min_side": args.min_side,
                    "max_frames_per_group": args.max_frames_per_group,
                    "jpeg_quality": args.jpeg_quality},
        "prompt": {"style_in_captions": args.prompt_style,
                   "skin_tone": args.skin_tone,
                   "upper_layer": args.upper_layer},
        "split": {"val_frac": args.val_frac, "seed": args.seed,
                  "train_actors": len(splits) - n_val, "val_actors": n_val,
                  "val_actor_ids": sorted(a for a, s in splits.items()
                                          if s == "val"),
                  "crops_per_split": dict(per_split)},
        "plan_counters": dict(counters),
        "crops_indexed": len(annotations),
        "crops_planned": counters.get("boxes_kept", 0),
        "crops_skipped_min_side": skipped,
        "crops_still_missing": missing,
        "crops_recovered_from_disk": recovered,
        "complete": missing == 0,
        "crops_per_altitude_m": dict(sorted(per_alt.items())),
        "crops_per_activity": dict(sorted(per_act.items())),
        "crops_per_actor": {"actors": len(per_actor),
                            "min": min(per_actor.values()) if per_actor else 0,
                            "max": max(per_actor.values()) if per_actor else 0},
        "crop_width_px": percentiles([r["crop_w"] for r in annotations]),
        "crop_height_px": percentiles([r["crop_h"] for r in annotations]),
        "unique_captions": len({r["captions"][0] for r in annotations}),
    }
    (out / "build_report.json").write_text(json.dumps(report, indent=2),
                                           encoding="utf-8")
    log.info("[finalize] %d crop(s) indexed (%d planned, %d recovered from "
             "disk, %d missing, %d below --min-side) -> %s", len(annotations),
             report["crops_planned"], recovered, missing, skipped, out)
    if missing:
        log.warning("[finalize] INCOMPLETE: %d planned crop(s) are not on disk. "
                    "Re-run without --finalize to continue; the build is "
                    "resumable and costs nothing for work already done.", missing)
    print(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nomad-root",
                    default=str(Path(__file__).resolve().parents[2]
                                / "Dataset" / "NOMAD"))
    ap.add_argument("--out", default=None, help="default: <nomad-root>/crop")
    ap.add_argument("--altitudes", nargs="+", default=["a10", "a30"],
                    choices=ALL_ALTITUDES,
                    help="which flight altitudes to crop (default a10 a30)")
    ap.add_argument("--min-visibility", type=int, default=50,
                    help="keep boxes whose NOMAD visibility label is >= this "
                         "(10..100, default 50)")
    ap.add_argument("--pad", type=float, default=0.12,
                    help="fraction of box w/h added on every side (default 0.12)")
    ap.add_argument("--min-side", type=int, default=24,
                    help="drop crops whose short side is under this many pixels")
    ap.add_argument("--max-frames-per-group", type=int, default=0,
                    help="cap frames per (actor, altitude); 0 = no cap")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of ACTORS held out for validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-style", default="default",
                    choices=["default", "templated", "templated_neg"],
                    help="which prompt flavour lands in the 'captions' field "
                         "(all three are always stored in the record)")
    ap.add_argument("--skin-tone", default="none", choices=["none", "coarse"],
                    help="'coarse' derives a skin-tone attribute from NOMAD's "
                         "self-reported race field (off by default)")
    ap.add_argument("--upper-layer", default="last", choices=["last", "first"],
                    help="15 actors list two upper garments. 'last' (default) "
                         "captions the OUTER one -- what a top-down drone view "
                         "actually shows; 'first' captions the base layer. The "
                         "other layer is kept as an other_feature clause either "
                         "way.")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be cropped, decode nothing")
    ap.add_argument("--limit-actors", type=int, default=0,
                    help="debug: only process the first N actors")
    # bounded / resumable execution
    ap.add_argument("--plan-cache", default=None, metavar="PATH",
                    help="pickle the crop plan here and reuse it next run. "
                         "Planning re-parses NOMAD's 9.7 MB annotations.json "
                         "(~10 s), which dominates a short pass. Delete the file "
                         "after changing any filter.")
    ap.add_argument("--time-budget", type=float, default=0, metavar="SEC",
                    help="decode for at most SEC seconds, journal progress, then "
                         "report how many frames remain")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="decode only shard I of N (0-based) of the plan")
    ap.add_argument("--order", default="interleave",
                    choices=["interleave", "plan"],
                    help="'interleave' (default) round-robins frames across "
                         "actors so a partially finished build is still a "
                         "balanced subset of all identities; 'plan' keeps "
                         "NOMAD's actor-by-actor file order")
    ap.add_argument("--finalize", action="store_true",
                    help="skip decoding; write annotations.json from the resume "
                         "journal. Run once after the last bounded pass.")
    ap.add_argument("--verify", action="store_true",
                    help="check every crop is a COMPLETE JPEG, delete the ones "
                         "that aren't (plus orphan .tmp files) and drop them "
                         "from the journal, so a normal re-run regenerates "
                         "them. Cheap (2 bytes per file). Run this after any "
                         "interrupted build -- a truncated crop passes "
                         "Image.open() and only fails mid-training.")
    args = ap.parse_args()
    args.out = args.out or str(Path(args.nomad_root) / "crop")

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    out = Path(args.out)

    # ---- verify-only, without the source annotations --------------------- #
    # A training server gets Dataset/NOMAD/crop and nothing else (the source
    # images are 344 GB). `--verify` is documented as the thing to run after
    # copying, so it must not require annotations.json to exist.
    ann = Path(args.nomad_root) / "annotations" / "annotations.json"
    if args.verify and not args.finalize and not ann.is_file():
        log.info("[verify] %s not found -- verifying the crops on disk only",
                 ann)
        raise SystemExit(verify_crops_standalone(out, args.workers))

    # ---- plan (cached) --------------------------------------------------- #
    cache = Path(args.plan_cache) if args.plan_cache else None
    if cache and cache.is_file():
        jobs, counters, captions, splits = pickle.loads(cache.read_bytes())
        log.info("[plan] reused %s: %d frame(s) -> %d crop(s)", cache,
                 len(jobs), counters["boxes_kept"])
    else:
        jobs, counters, captions, splits = build_plan(args)
        log.info("[plan] %d frame(s) to decode -> %d crop(s); %s",
                 len(jobs), counters["boxes_kept"], counters)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps((jobs, counters, captions, splits)))
            log.info("[plan] cached to %s", cache)

    if args.dry_run:
        n_val = sum(1 for v in splits.values() if v == "val")
        print(json.dumps({"frames_to_decode": len(jobs),
                          "crops_to_write": counters["boxes_kept"],
                          "counters": counters,
                          "train_actors": len(splits) - n_val,
                          "val_actors": n_val}, indent=2))
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    progress_path = out / PROGRESS_NAME
    done = read_progress(progress_path)
    log.info("[resume] %d crop(s) already journalled", len(done))

    if args.verify:
        done = verify_crops(out, jobs, done, args.workers)
        if not args.finalize:
            return

    if args.finalize:
        finalize(args, out, jobs, counters, captions, splits, done)
        return

    # ---- work selection -------------------------------------------------- #
    todo = [j for j in jobs
            if any(b["tag"] not in done for b in j["boxes"])]
    if args.order == "interleave":
        # NOMAD's annotation file is ordered Actor001, Actor002, ... so plan
        # order would finish actor-by-actor: an interrupted build then covers
        # the first N actors and none of the rest, which is useless as a
        # training set. Round-robin across actors instead, so ANY partial state
        # is a balanced subset of all 100 identities.
        by_actor: dict = defaultdict(list)
        for j in todo:
            by_actor[j["actor"]].append(j)
        actors = sorted(by_actor)
        todo = [j for row in zip_longest(*(by_actor[a] for a in actors))
                for j in row if j is not None]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"--shard {args.shard}: need 0 <= I < N")
        todo = todo[i::n]
        log.info("[shard] %d/%d -> %d frame(s) this pass", i, n, len(todo))
    total_todo = len(todo)
    log.info("[work] %d frame(s) need decoding", total_todo)

    # ---- decode ---------------------------------------------------------- #
    errors: list[str] = []
    decoded = 0
    t0 = time.time()
    with open(progress_path, "a", encoding="utf-8") as fh:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            inflight = []
            it = iter(todo)
            exhausted = False
            while True:
                # keep the queue shallow so the time budget stays responsive
                while not exhausted and len(inflight) < args.workers * 3:
                    if args.time_budget and time.time() - t0 > args.time_budget:
                        exhausted = True
                        break
                    try:
                        inflight.append(ex.submit(crop_one_frame,
                                                  to_task(next(it))))
                    except StopIteration:
                        exhausted = True
                if not inflight:
                    break
                for fut in as_completed(inflight, timeout=None):
                    status, err, sizes = fut.result()
                    inflight.remove(fut)
                    decoded += 1
                    if err:
                        errors.append(err)
                    append_progress(fh, sizes)
                    done.update(sizes)
                    if decoded % 200 == 0:
                        log.info("[crop] %d/%d frame(s), %.0fs elapsed",
                                 decoded, total_todo, time.time() - t0)
                    break        # re-check budget / refill the queue

    left = total_todo - decoded
    log.info("[crop] %d frame(s) decoded in %.0fs (%.1f frames/s), %d left, "
             "%d error(s)", decoded, time.time() - t0,
             decoded / max(time.time() - t0, 1e-6), max(0, left), len(errors))
    if errors:
        log.warning("[crop] first errors: %s", errors[:5])

    if left > 0 and (args.time_budget or args.shard):
        # A bounded pass must not write annotations.json: it would describe a
        # partial dataset. Journal only; finalize when the work is done.
        print(json.dumps({"decoded_frames": decoded, "frames_left": left,
                          "journalled_crops": len(done),
                          "errors": len(errors), "errors_sample": errors[:5]},
                         indent=2))
        return

    finalize(args, out, jobs, counters, captions, splits, done)


if __name__ == "__main__":
    main()
