"""Full-set retrieval benchmarks with identity labels (the Tier-1/2/3 data layer).

Why this module exists
----------------------
``train_dataset.load_benchmark_pairs`` builds an *n x n* evaluation matrix: one
caption per image, one correct column per row, gallery subsampled to
``--eval-on-size``. That is fine as a training-time progress signal, but it is
**not** the protocol the TBPS literature reports, so its numbers cannot be put
next to IRRA / RDE / APTM. Three differences matter:

1. **Gallery size.** The published protocol ranks every query against the FULL
   test gallery (ICFG-PEDES: 19 848 images). A 1 000-image gallery inflates R@1.
2. **Identity ground truth.** A caption is correct for *every* image of the same
   person, not only for the one image it was written from. Without pids you can
   compute R@K, but mAP/mINP are undefined (or, worse, silently wrong).
3. **Query set.** Every test caption is a query (RSTPReid: 2 captions per image
   -> 2 000 queries over a 1 000-image gallery).

This module produces an :class:`EvalSet` -- ``gallery`` (images + pid) and
``queries`` (captions + pid) -- and caches it as a JSON manifest so that every
run, every baseline and every occluded variant is scored on byte-identical data.

Available benchmarks
--------------------
``icfg``   ICFG-PEDES test split, from ``ICFG-PEDES.json``.
           19 848 images / 1 000 ids / 19 848 captions -- matches the published
           protocol exactly.
``rstp``   RSTPReid test split, from the HF parquet (has ``id`` + ``split``).
           1 000 images / 200 ids / 2 000 captions.
``cuhk``   CUHK-PEDES test split. **Requires the official ``reid_raw.json``**;
           the HF parquet in ``Dataset/CUHK-PEDES/data`` carries only the TRAIN
           split and has neither ``id`` nor ``split`` columns (see
           :func:`_load_cuhk` for the exact error and how to fix it).
``nomad``  NOMAD aerial crops, actor-disjoint ``val`` split. Gallery = held-out
           crops, one query per held-out actor. Carries ``altitude_m`` /
           ``visibility`` / ``activity`` per gallery image so Tier-3 tables can
           be broken down by flight altitude and real occlusion level.

CLI
---
    # build + cache every manifest that this machine has the data for
    python -m sar_person.eval_sets --build icfg rstp nomad \
        --data-root ../Dataset --out ../Dataset/Result/evalsets

    # inspect one without building
    python -m sar_person.eval_sets --describe icfg --data-root ../Dataset
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)

MANIFEST_VERSION = 1


# --------------------------------------------------------------------------- #
# Container                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class EvalSet:
    """One frozen retrieval benchmark.

    gallery : [{"image": abs path, "pid": str, "image_id": str, **meta}]
    queries : [{"text": str, "pid": str, "image_id": str}]
    """
    name: str
    gallery: List[Dict]
    queries: List[Dict]
    meta: Dict = field(default_factory=dict)

    # -- derived views -----------------------------------------------------
    @property
    def gallery_paths(self) -> List[str]:
        return [g["image"] for g in self.gallery]

    @property
    def query_texts(self) -> List[str]:
        return [q["text"] for q in self.queries]

    @property
    def gallery_pids(self):
        import numpy as np
        return np.asarray([str(g["pid"]) for g in self.gallery])

    @property
    def query_pids(self):
        import numpy as np
        return np.asarray([str(q["pid"]) for q in self.queries])

    def stats(self) -> Dict:
        gp, qp = {g["pid"] for g in self.gallery}, {q["pid"] for q in self.queries}
        orphan = sorted(qp - gp)
        return {
            "gallery_images": len(self.gallery),
            "gallery_ids": len(gp),
            "queries": len(self.queries),
            "query_ids": len(qp),
            "queries_without_gallery_match": len(orphan),
            "avg_gallery_per_query_id": (
                round(len(self.gallery) / max(len(qp), 1), 2)),
        }

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": MANIFEST_VERSION,
            "name": self.name,
            "meta": self.meta,
            "stats": self.stats(),
            "gallery": self.gallery,
            "queries": self.queries,
        }, indent=1), encoding="utf-8")
        log.info("[evalset] wrote %s (%s)", path, self.stats())
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EvalSet":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("version") != MANIFEST_VERSION:
            log.warning("[evalset] manifest version %s != %s (%s)",
                        d.get("version"), MANIFEST_VERSION, path)
        return cls(d["name"], d["gallery"], d["queries"], d.get("meta", {}))

    def with_gallery_images(self, mapping: Dict[str, str], name: str,
                            extra_meta: Dict | None = None) -> "EvalSet":
        """Return a copy whose gallery paths are remapped (Tier-2 occlusion).

        Queries and pids are untouched by construction: the text still describes
        the whole person, only the image is degraded -- exactly the SAR case.
        Gallery entries with no remapped image are dropped, so a partial
        degradation run still yields a self-consistent (smaller) benchmark.
        """
        gal = []
        for g in self.gallery:
            new = mapping.get(g["image"])
            if new:
                gal.append({**g, "image": new, "source_image": g["image"]})
        meta = {**self.meta, "derived_from": self.name, **(extra_meta or {})}
        return EvalSet(name, gal, list(self.queries), meta)


# --------------------------------------------------------------------------- #
# Loaders                                                                      #
# --------------------------------------------------------------------------- #
def _load_icfg(root: Path, truncate: int | None = None, name: str = "icfg",
               **_) -> EvalSet:
    """ICFG-PEDES test split.

    ``truncate`` reproduces MGCC's ``process_data.py``::

        elif (split == 'test' and (len(test_json) < 10000)):
            test_json.append(data_save)
        else:                        # the remainder is pushed into val
            val_json.append(data_save)

    i.e. the *first* N test records in the annotation file's own order -- not a
    random sample and not identity-stratified. Since ICFG's JSON is grouped by
    identity, truncating at 10 000 keeps roughly the first half of the
    identities and drops the rest entirely, so the MGCC gallery is smaller in
    both images *and* ids. Both effects raise R@1, which is why MGCC's 59.28 on
    Occluded-ICFG cannot be read against a 19 848-image number.

    Kept as a separate benchmark rather than a flag so the two manifests coexist
    on disk and every result file records which gallery it was scored on.
    """
    ann = root / "ICFG-PEDES" / "ICFG-PEDES" / "ICFG-PEDES.json"
    imgs = root / "ICFG-PEDES" / "ICFG-PEDES" / "imgs"
    if not ann.is_file():
        raise FileNotFoundError(f"ICFG annotation not found: {ann}")
    recs = [r for r in json.loads(ann.read_text(encoding="utf-8"))
            if str(r.get("split")) == "test"]
    n_full = len(recs)
    if truncate:
        # truncate BEFORE the on-disk existence check, exactly as MGCC does --
        # they count records, not files
        recs = recs[:truncate]
    gallery, queries, missing = [], [], 0
    for r in recs:
        p = imgs / r["file_path"]
        if not p.exists():
            missing += 1
            continue
        pid = str(r["id"])
        iid = r["file_path"]
        gallery.append({"image": str(p), "pid": pid, "image_id": iid})
        for c in r.get("captions") or []:
            if c and c.strip():
                queries.append({"text": c.strip(), "pid": pid, "image_id": iid})
    if missing:
        log.warning("[evalset:%s] %d test image(s) missing on disk",
                    name, missing)
    meta = {"dataset": "ICFG-PEDES", "split": "test", "source": str(ann)}
    if truncate:
        n_ids = len({g["pid"] for g in gallery})
        log.warning(
            "[evalset:%s] MGCC-style truncation: %d of %d test record(s), "
            "%d of 1000 identity/identities. This is NOT the published ICFG "
            "protocol -- use it only to sit next to MGCC's numbers, and say so "
            "in the table caption.", name, len(recs), n_full, n_ids)
        meta.update({"protocol": "mgcc", "truncated_to": truncate,
                     "n_test_records_full": n_full,
                     "gallery_ids": n_ids,
                     "not_comparable_to": "published ICFG-PEDES tables"})
    else:
        meta["expected"] = {"gallery_images": 19848, "gallery_ids": 1000,
                            "queries": 19848}
    return EvalSet(name, gallery, queries, meta)


def _load_icfg_mgcc(root: Path, **kw) -> EvalSet:
    kw.pop("truncate", None)
    kw.pop("name", None)
    return _load_icfg(root, truncate=10_000, name="icfg_mgcc", **kw)


def _load_forestperson(root: Path, **_) -> EvalSet:
    """ForestPerson test split -- small, but the only ground-level set here with
    a *real* per-image occlusion label.

    ``fields.visible_ratio`` (100 / 70 / 40 / 20) is annotated, not synthesised.
    That makes this the cheapest available answer to the reviewer question the
    Tier-2 sweep cannot answer on its own: "does your synthetic occlusion behave
    like real occlusion?" Run the same model over the visibility buckets here
    and over the coverage levels of `ogor`, and report whether the two curves
    agree. It is exposed as ``visibility`` so ``--breakdown visibility`` -- the
    same flag NOMAD uses -- works unchanged.

    Size caveat, and it is not a small one: 168 gallery images over **14
    identities**. R@1 moves in steps of ~0.6 points, and with 14 identities a
    random ranker already scores well, so treat this as a domain/robustness
    probe, never as a headline retrieval benchmark. Report mAP and mINP
    alongside, and always print the identity count in the caption.
    """
    ann = root / "ForestPerson" / "forestperson_reid.json"
    if not ann.is_file():
        raise FileNotFoundError(f"ForestPerson annotation not found: {ann}")
    gallery, queries, missing = [], [], 0
    for r in json.loads(ann.read_text(encoding="utf-8")):
        if str(r.get("split")) != "test":
            continue
        p = root / r["file_path"]
        if not p.exists():
            missing += 1
            continue
        f = r.get("fields") or {}
        pid, iid = str(r["id"]), r["file_path"]
        entry = {"image": str(p), "pid": pid, "image_id": iid}
        vis = f.get("visible_ratio")
        if vis is not None:
            try:
                entry["visibility"] = float(vis)
            except (TypeError, ValueError):
                pass
        for k in ("pose", "place", "season", "weather", "scene"):
            if f.get(k):
                entry[k] = f[k]
        gallery.append(entry)
        for c in r.get("captions") or []:
            if c and c.strip():
                queries.append({"text": c.strip(), "pid": pid, "image_id": iid})
    if missing:
        log.warning("[evalset:forestperson] %d test image(s) missing on disk",
                    missing)
    n_ids = len({g["pid"] for g in gallery})
    if n_ids < 50:
        log.warning("[evalset:forestperson] only %d identity/identities in the "
                    "gallery -- R@1 is coarse and optimistic here. Use this for "
                    "the visibility breakdown, not for a SOTA table.", n_ids)
    return EvalSet("forestperson", gallery, queries, {
        "dataset": "ForestPerson", "split": "test", "source": str(ann),
        "domain": "forest, ground-level",
        "occlusion": "real, annotated (visible_ratio)",
        "breakdown_fields": ["visibility", "pose", "place", "season"],
        "gallery_ids": n_ids,
        "not_comparable_to": "published TBPS tables (no published protocol)",
    })


def _load_rstp(root: Path, image_out: Path | None = None, **_) -> EvalSet:
    import pyarrow.parquet as pq
    pdir = root / "RSTPReid" / "data"
    out = Path(image_out or (root / "RSTPReid" / "images"))
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(str(pdir / "**" / "*.parquet"), recursive=True))
    if not shards:
        raise FileNotFoundError(f"no parquet under {pdir}")
    gallery, queries = [], []
    for sp in shards:
        stem = Path(sp).stem
        for i, row in enumerate(pq.read_table(sp).to_pylist()):
            if str(row.get("split")) != "test":
                continue
            cell = row.get("image") or {}
            data = cell.get("bytes") if isinstance(cell, dict) else cell
            if not data:
                continue
            name = row.get("file_path") or f"{stem}_{i}.jpg"
            fp = out / f"{stem}__{Path(str(name)).name}"
            if not fp.exists():
                fp.write_bytes(data)
            pid = str(row.get("id"))
            iid = str(row.get("file_path") or fp.name)
            gallery.append({"image": str(fp), "pid": pid, "image_id": iid})
            for c in row.get("captions") or []:
                if c and str(c).strip():
                    queries.append({"text": str(c).strip(), "pid": pid,
                                    "image_id": iid})
    return EvalSet("rstp", gallery, queries, {
        "dataset": "RSTPReid", "split": "test", "source": str(pdir),
        "expected": {"gallery_images": 1000, "gallery_ids": 200,
                     "queries": 2000},
    })


_CUHK_HINT = """
CUHK-PEDES test split is NOT available from the parquet shards in this project.

  Dataset/CUHK-PEDES/data/*.parquet  ->  columns: image{bytes,path}, text
                                         shards : train-*  (TRAIN split only)
                                         238 768 rows / 34 052 images / 11 002 ids
                                         == the official TRAIN split, captions
                                            augmented to 4-8 per image (official
                                            has 2)

There is no `id` column and no `split` column, which means:
  * the split filter is a NO-OP -> `--eval-on cuhk` in train_dataset has been
    scoring the TRAINING split. Treat every past CUHK number as invalid.

To enable Tier-1 on CUHK-PEDES, place the official annotation file at

    Dataset/CUHK-PEDES/reid_raw.json

(list of {split, file_path, id, captions}), with the images under

    Dataset/CUHK-PEDES/imgs/<file_path>

That file ships with the original CUHK-PEDES release (Li et al., CVPR 2017) and
is what IRRA / RDE / APTM consume, so using it is what makes your numbers
comparable to their tables. Pass --cuhk-json / --cuhk-images to override paths.

If you have the author's release but NOT reid_raw.json (the 2017 zip ships
`caption_all.json`, which has no `split` field), run

    python scripts/prepare_cuhk.py

It recovers the official TRAIN split exactly and RECONSTRUCTS val/test from the
2 000 remaining identities, writing reid_raw.json plus a cuhk_provenance.json
that marks the test split as reconstructed -- this loader then reports the
observed shape instead of pretending to match the literature.

Where to get it (per the authors' repo,
github.com/ShuangLI59/Person-Search-with-Natural-Language-Description):
  * reid_raw.json  -- linked publicly from that README (Google Drive)
  * the IMAGES     -- request from  tong.xiao.work@gmail.com  (academic only;
                      the authors ask that you write from a university address)

If you only have the train shards, `--build cuhk_heldout` carves an
identity-disjoint held-out set out of them. Read that builder's docstring first:
its numbers are usable for Tier 2/3, and are NOT comparable to any published
CUHK-PEDES table.
""".strip()


# --------------------------------------------------------------------------- #
# CUHK-PEDES identity recovery                                                 #
# --------------------------------------------------------------------------- #
# CUHK-PEDES is a union of five re-ID sources, each keeping its own filename
# convention, and the pid is encoded in the filename. Those pid namespaces
# OVERLAP (Market pid 339 and VIPeR pid 339 are different people), so the parsed
# label is always prefixed with the source -- merging them would silently fuse
# identities and inflate every metric.
_CUHK_SOURCES = (
    ("cuhk01", re.compile(r"^(\d{4})\d{3}\.png$")),          # 0363001.png
    ("market", re.compile(r"^(-?\d+)_c\ds\d_\d+_\d+\.jpg$")),  # 1407_c3s3_049303_00.jpg
    ("sysu",   re.compile(r"^p(\d+)_s\d+\.jpg$")),            # p10082_s15238.jpg
    ("viper",  re.compile(r"^(\d+)_\d+\.bmp$")),              # 339_90.bmp
    ("cuhk03", re.compile(r"^(\d+)_\d+\.jpg$")),              # 1179_1.jpg
)


def parse_cuhk_pid(filename: str) -> str | None:
    """'1407_c3s3_049303_00.jpg' -> 'market:1407'. None if unrecognised.

    A handful of files in the HF export carry a doubled extension
    ('...jpg.jpg'); stripping it recovers a valid Market name rather than
    dropping four images on the floor.
    """
    name = Path(str(filename).replace("\\", "/")).name
    for _ in range(2):                       # unwrap at most one doubled suffix
        for src, pat in _CUHK_SOURCES:
            m = pat.match(name)
            if m:
                return f"{src}:{m.group(1)}"
        if name.endswith((".jpg.jpg", ".png.png", ".bmp.bmp")):
            name = name[: -len(Path(name).suffix)]
        else:
            break
    return None


def _load_cuhk(root: Path, cuhk_json: str | None = None,
               cuhk_images: str | None = None, **_) -> EvalSet:
    ann = Path(cuhk_json) if cuhk_json else (root / "CUHK-PEDES" / "reid_raw.json")
    imgs = Path(cuhk_images) if cuhk_images else (root / "CUHK-PEDES" / "imgs")
    if not ann.is_file():
        raise FileNotFoundError(f"{ann} not found.\n\n{_CUHK_HINT}")
    recs = [r for r in json.loads(ann.read_text(encoding="utf-8"))
            if str(r.get("split")) == "test"]
    gallery, queries, missing = [], [], 0
    for r in recs:
        p = imgs / r["file_path"]
        if not p.exists():
            missing += 1
            continue
        pid, iid = str(r["id"]), r["file_path"]
        gallery.append({"image": str(p), "pid": pid, "image_id": iid})
        for c in r.get("captions") or []:
            if c and c.strip():
                queries.append({"text": c.strip(), "pid": pid, "image_id": iid})
    if missing:
        log.warning("[evalset:cuhk] %d test image(s) missing under %s",
                    missing, imgs)
    official = {"gallery_images": 3074, "gallery_ids": 1000, "queries": 6156}
    meta = {"dataset": "CUHK-PEDES", "split": "test", "source": str(ann),
            "split_source": "official", "expected": official}
    # `prepare_cuhk.py` can only recover the official TRAIN split; val/test are
    # reconstructed from the 2 000 remaining identities. Warning about the
    # 4-image difference from the published gallery on every single run would
    # train the reader to ignore the warning -- so say it once, in the metadata,
    # where the results table can pick it up.
    prov = ann.parent / "cuhk_provenance.json"
    if prov.is_file():
        try:
            p = json.loads(prov.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            p = {}
        status = ((p.get("splits") or {}).get("val/test") or {}).get("status", "")
        if status.startswith("RECONSTRUCTED"):
            meta["split_source"] = "reconstructed"
            meta["official_expected"] = official
            meta["expected"] = {
                "gallery_images": len(gallery),
                "gallery_ids": len({g["pid"] for g in gallery}),
                "queries": len(queries)}
            log.warning("[evalset:cuhk] test split is RECONSTRUCTED, not the "
                        "authors' -- report it as such, do not put it in a "
                        "table next to IRRA/RDE/APTM without a footnote "
                        "(%s)", prov)
    return EvalSet("cuhk", gallery, queries, meta)


def _load_nomad(root: Path, nomad_prompt_style: str = "default",
                nomad_split: str = "val", nomad_crop: str = "crop",
                nomad_min_vis: int = 0, **_) -> EvalSet:
    """NOMAD actors. Gallery = every crop in the split; queries = one per actor.

    NOMAD stores ONE description per actor, so the query set is small (20 actors
    in the default 80/20 actor split). That is a faithful SAR setting -- one
    missing-person profile against a large ROI gallery -- but it means R@1 moves
    in 1/20 steps. Report it with the query count in the caption, and lean on
    mAP / mINP (which use the whole ranking) rather than on R@1 alone.

    ``nomad_split="all"`` uses every actor, which takes the R@1 step from 5
    points to 1 and is what a per-visibility breakdown needs -- ten buckets of a
    20-query set cannot carry a curve. It is only honest for a model that did
    NOT train on NOMAD; on a NOMAD-trained adapter it reports memorisation.

    ``nomad_crop`` picks the prepared crop directory under ``NOMAD/``. This is
    load-bearing for the occlusion study: the default build (``crop``) was made
    with ``--min-visibility 50``, so it physically CANNOT show occlusion beyond
    50% -- those 7 843 crops were never written to disk. A real-occlusion curve
    needs a set prepared with ``--min-visibility 10`` (e.g. ``crop_all``).

    ``nomad_min_vis`` tightens the floor at eval time. It can only drop crops
    the prepared set already contains; it can never recover the ones the
    prepare-time filter discarded.
    """
    crop = root / "NOMAD" / nomad_crop
    ann = crop / "annotations.json"
    if not ann.is_file():
        raise FileNotFoundError(
            f"{ann} not found -- run scripts/prepare_nomad.py --out {crop} "
            f"first (--min-visibility 10 if you need the occluded buckets)")
    try:
        key = NOMAD_PROMPT_KEYS[nomad_prompt_style]
    except KeyError:
        raise ValueError(
            f"unknown nomad_prompt_style {nomad_prompt_style!r}; "
            f"have {sorted(NOMAD_PROMPT_KEYS)}") from None
    all_recs = json.loads(ann.read_text(encoding="utf-8"))
    recs = [r for r in all_recs
            if nomad_split == "all" or str(r.get("split")) == nomad_split]
    if not recs:
        raise ValueError(
            f"nomad_split={nomad_split!r} matched 0 of {len(all_recs)} record(s) "
            f"in {ann}; splits present: "
            f"{sorted({str(r.get('split')) for r in all_recs})} (or 'all')")
    if nomad_min_vis:
        recs = [r for r in recs
                if int(r.get("visibility") or 0) >= int(nomad_min_vis)]
    imgs = crop / "images"
    gallery, seen_actor, queries, missing, fallback = [], {}, [], 0, 0
    for r in recs:
        p = imgs / r["file_path"]
        if not p.exists():
            missing += 1
            continue
        pid = str(r["person_id"])
        gallery.append({
            "image": str(p), "pid": pid, "image_id": str(r["id"]),
            "altitude_m": r.get("altitude_m"), "visibility": r.get("visibility"),
            "activity": r.get("activity"), "environment": r.get("environment"),
        })
        if pid not in seen_actor:
            txt = r.get(key)
            if not txt:
                # B1: a silent fall-back to captions[0] makes a missing prompt
                # flavour look exactly like the cache bug this patch fixes --
                # every style would produce the same query text and the table
                # would show four identical rows. Count it, fail loudly below.
                fallback += 1
                txt = (r.get("captions") or [""])[0]
            if txt:
                seen_actor[pid] = txt
    for pid, txt in sorted(seen_actor.items()):
        queries.append({"text": txt, "pid": pid, "image_id": f"actor_{pid}"})
    if missing:
        log.warning("[evalset:nomad] %d crop(s) missing on disk", missing)
    if fallback:
        raise RuntimeError(
            f"[evalset:nomad] {fallback} record(s) have no {key!r} field -- "
            f"prompt style {nomad_prompt_style!r} is NOT actually available in "
            f"annotations.json. Scoring it would silently reuse captions[0] and "
            f"produce a row identical to another style. Regenerate the "
            f"annotations (scripts/prepare_nomad.py) or drop this style.")
    vis_levels = sorted({int(g["visibility"]) for g in gallery
                         if g.get("visibility") is not None})
    log.info("[evalset:nomad] crop=%s split=%s | %d gallery, %d queries | "
             "visibility levels present: %s", nomad_crop, nomad_split,
             len(gallery), len(queries), vis_levels or "none")
    if vis_levels:
        # The occlusion curve can only reach 1 - min(visibility)/100. Say where
        # it stops, because the ceiling is a property of the PREPARED SET, not
        # of the data: prepare_nomad's --min-visibility filter deleted the rest,
        # and no eval flag can bring those crops back.
        log.info("[evalset:nomad] occlusion coverage available: 0.00 .. %.2f "
                 "(visibility %d%%..%d%%). Crops below %d%% visibility were "
                 "dropped at prepare time; rerun scripts/prepare_nomad.py with "
                 "a lower --min-visibility to extend the curve.",
                 1 - min(vis_levels) / 100.0, min(vis_levels), max(vis_levels),
                 min(vis_levels))
    return EvalSet("nomad", gallery, queries, {
        "dataset": "NOMAD", "split": nomad_split, "source": str(ann),
        "prompt_style": nomad_prompt_style,
        "crop_dir": nomad_crop, "min_vis": int(nomad_min_vis),
        "visibility_levels": vis_levels,
        "note": "one query per actor; gallery keeps altitude/visibility so "
                "Tier-3 can be broken down. visibility is NOMAD's own "
                "annotation (% of body inside the box), so occlusion = "
                "1 - visibility/100 is REAL, not synthetic",
    })


def _load_cuhk_heldout(root: Path, heldout_ids: int = 1000,
                       max_captions_per_image: int = 2, seed: int = 0,
                       image_out: str | None = None, **_) -> EvalSet:
    """An identity-disjoint held-out set carved out of the CUHK-PEDES TRAIN shards.

    ⚠️  NOT COMPARABLE TO ANY PUBLISHED CUHK-PEDES NUMBER. Read this before using
    it in a table.

    What it is. The parquet in ``Dataset/CUHK-PEDES/data`` is exactly the
    official *train* split (34 052 images / 11 002 ids, versus the official
    34 054 / 11 003) with captions augmented to 4-8 per image. pids are not
    stored, but they are recoverable from the filenames (see
    :func:`parse_cuhk_pid`), which is what makes this possible at all.

    This builder sorts the identities, takes ``heldout_ids`` of them with a fixed
    seed, and turns them into a gallery + query set sized like the official test
    split (1 000 ids, ~3 000 images, ~2 captions per image).

    Why the numbers are not comparable, in order of severity:

    1. **Different gallery.** Published R@1 is measured against the specific
       3 074 images of Li et al.'s test split. Rank-based metrics move with
       gallery composition and size, so a different 1 000 ids is a different
       number, not a noisy version of the same one.
    2. **The baselines have seen these people.** Every public IRRA / RDE / APTM
       checkpoint was trained on the full CUHK-PEDES train split -- i.e. on these
       identities. Scoring them here measures memorisation; scoring your model
       here measures generalisation. The comparison is rigged against you.
    3. **Captions are augmented.** 4-8 per image instead of the official 2, from
       an unknown generator. ``max_captions_per_image`` caps the query set at 2
       to keep its scale sane, but they are still not the official sentences.

    What it IS good for: Tier 2 (occlusion robustness) and Tier 3 (domain
    transfer), where there are no published numbers anyway and every baseline is
    re-run by you under one protocol. Call it "CUHK-PEDES-train held-out" in the
    paper, never "CUHK-PEDES".

    Leakage. These identities must not appear in training. The builder writes
    ``<manifest>.excluded_images.json`` next to the manifest; pass it to
    ``train_dataset --exclude-images`` so the held-out images are dropped from
    the training pairs.
    """
    import pyarrow.parquet as pq
    pdir = root / "CUHK-PEDES" / "data"
    out = Path(image_out or (root / "CUHK-PEDES" / "images"))
    shards = sorted(glob.glob(str(pdir / "**" / "*.parquet"), recursive=True))
    if not shards:
        raise FileNotFoundError(f"no parquet under {pdir}\n\n{_CUHK_HINT}")

    # pass 1: filenames only, to choose the held-out identities before decoding
    # a single image (decoding all 34k would cost minutes and ~4 GB for nothing)
    by_pid: Dict[str, set] = {}
    unparsed = 0
    for sp in shards:
        for b in pq.ParquetFile(sp).iter_batches(batch_size=4096,
                                                 columns=["image"]):
            for cell in b.to_pydict()["image"]:
                nm = (cell or {}).get("path")
                if not nm:
                    continue
                pid = parse_cuhk_pid(nm)
                if pid is None:
                    unparsed += 1
                    continue
                by_pid.setdefault(pid, set()).add(nm)
    if not by_pid:
        raise RuntimeError(f"no identity could be parsed from {pdir}")
    if unparsed:
        log.warning("[evalset:cuhk_heldout] %d filename(s) with an unrecognised "
                    "pattern were skipped", unparsed)

    import random as _random
    pids = sorted(by_pid)
    rng = _random.Random(seed)
    chosen = set(rng.sample(pids, min(heldout_ids, len(pids))))
    wanted = {nm: pid for pid in chosen for nm in by_pid[pid]}
    log.info("[evalset:cuhk_heldout] %d/%d identities, %d image(s), seed=%d",
             len(chosen), len(pids), len(wanted), seed)

    # pass 2: materialise only the chosen images and collect their captions
    out.mkdir(parents=True, exist_ok=True)
    gallery, queries = [], []
    caps_seen: Dict[str, int] = {}
    written = set()
    for sp in shards:
        stem = Path(sp).stem
        for b in pq.ParquetFile(sp).iter_batches(batch_size=2048,
                                                 columns=["image", "text"]):
            d = b.to_pydict()
            for cell, txt in zip(d["image"], d["text"]):
                nm = (cell or {}).get("path")
                pid = wanted.get(nm)
                if pid is None:
                    continue
                fp = out / f"{stem}__{Path(str(nm)).name}"
                if nm not in written:
                    data = (cell or {}).get("bytes")
                    if not data:
                        continue
                    if not fp.exists():
                        fp.write_bytes(data)
                    gallery.append({"image": str(fp), "pid": pid,
                                    "image_id": nm, "source": pid.split(":")[0]})
                    written.add(nm)
                t = (txt or "").strip()
                if t and caps_seen.get(nm, 0) < max_captions_per_image:
                    caps_seen[nm] = caps_seen.get(nm, 0) + 1
                    queries.append({"text": t, "pid": pid, "image_id": nm})

    return EvalSet("cuhk_heldout", gallery, queries, {
        "dataset": "CUHK-PEDES (train shards, identity-disjoint holdout)",
        "split": f"self-made holdout of {len(chosen)} ids, seed={seed}",
        "source": str(pdir),
        "heldout_ids": sorted(chosen),
        "max_captions_per_image": max_captions_per_image,
        "NOT_COMPARABLE": "different gallery from the official test split, and "
                          "public IRRA/RDE/APTM checkpoints were trained on "
                          "these identities -- do not place these numbers next "
                          "to a published CUHK-PEDES table",
    })


BUILDERS = {"icfg": _load_icfg, "icfg_mgcc": _load_icfg_mgcc,
            "rstp": _load_rstp, "cuhk": _load_cuhk,
            "cuhk_heldout": _load_cuhk_heldout, "nomad": _load_nomad,
            "forestperson": _load_forestperson}

#: Benchmarks that do NOT follow the protocol the literature reports on. Results
#: from these belong in their own table, with the reason in the caption.
NONSTANDARD = {"icfg_mgcc": "MGCC's 10 000-record truncation of ICFG test",
               "cuhk_heldout": "self-made identity split of the CUHK train shards",
               "forestperson": "no published protocol; only 14 test identities"}

#: Training corpora that are deliberately NOT benchmarks.
#:
#: A retrieval benchmark needs identity labels: the t2i protocol scores a
#: gallery image as correct when it shares a pid with the query, so mAP and mINP
#: are only defined when an identity spans several images. MALS gives one
#: caption per image and no identity at all -- every image would be its own
#: identity, mAP would collapse to 1/rank (i.e. MRR) and mINP to a constant 1.0.
#: Those are not the metrics the literature reports under those names, so MALS
#: is a training set here and nothing else. Same for ForestSynth.
NOT_BENCHMARKS = {
    "mals": "one caption per image, no identity labels -- mAP degenerates to "
            "MRR and mINP to 1.0. Train on it (train_dataset --datasets mals).",
    "forestsynth": "synthetic per-person crops without a held-out identity "
                   "split. Train on it, evaluate on forestperson/nomad.",
}


# --------------------------------------------------------------------------- #
# Public entry point                                                           #
# --------------------------------------------------------------------------- #
# --- B1 patch: prompt-style-aware manifest cache ---
NOMAD_PROMPT_KEYS = {
    "default": "prompt_default",
    "templated": "prompt_templated",
    "templated_neg": "prompt_templated_neg",
    "contrastive": "prompt_contrastive",
}


def _cache_name(name: str, kw: dict) -> str:
    """Manifest filename. Anything that changes the QUERIES must appear here.

    ``(default, val)`` deliberately keeps the bare ``nomad.json`` name so the
    manifests already on disk -- and every number already measured against them
    -- remain valid.
    """
    if name != "nomad":
        return name
    style = str(kw.get("nomad_prompt_style") or "default")
    split = str(kw.get("nomad_split") or "val")
    crop = str(kw.get("nomad_crop") or "crop")
    mv = int(kw.get("nomad_min_vis") or 0)
    if (style, split, crop, mv) == ("default", "val", "crop", 0):
        return name
    tail = f"{name}__{style}__{split}"
    # The crop directory changes the GALLERY, not the queries, but it changes it
    # completely: crop_all holds 17 130 crops down to 10% visibility where crop
    # holds 9 287 down to 50%. Sharing one manifest between them would score two
    # different benchmarks under one name.
    if crop != "crop":
        tail += f"__{crop}"
    if mv:
        tail += f"__vis{mv}"
    return tail


def _cache_is_stale(es: "EvalSet", name: str, kw: dict) -> str:
    """'' if the cached manifest matches the request, else why it does not.

    _cache_name() already separates the files; this is the second line of
    defence, for the case where a manifest was written by an older build or
    hand-copied. Returning a REASON (not a bool) so the log says what differs.
    """
    meta = es.meta or {}
    if name == "nomad":
        want_style = str(kw.get("nomad_prompt_style") or "default")
        got_style = str(meta.get("prompt_style") or "default")
        if want_style != got_style:
            return f"prompt_style cached={got_style!r} requested={want_style!r}"
        want_split = str(kw.get("nomad_split") or "val")
        got_split = str(meta.get("split") or "val")
        if want_split != got_split:
            return f"split cached={got_split!r} requested={want_split!r}"
        want_crop = str(kw.get("nomad_crop") or "crop")
        got_crop = str(meta.get("crop_dir") or "crop")
        if want_crop != got_crop:
            return f"crop_dir cached={got_crop!r} requested={want_crop!r}"
        want_mv = int(kw.get("nomad_min_vis") or 0)
        got_mv = int(meta.get("min_vis") or 0)
        if want_mv != got_mv:
            return f"min_vis cached={got_mv} requested={want_mv}"
    return ""


def build(name: str, data_root: str | Path, cache_dir: str | Path | None = None,
          rebuild: bool = False, **kw) -> EvalSet:
    """Build (or load from cache) the named benchmark.

    Caching is on by default because materialising RSTPReid's parquet images and
    stat-ing ~20k ICFG paths is slow enough to discourage re-running the eval,
    and because a cached manifest is the only way to *prove* that two runs were
    scored on identical data.
    """
    if name in NOT_BENCHMARKS:
        raise ValueError(
            f"{name!r} is a training corpus, not a retrieval benchmark: "
            f"{NOT_BENCHMARKS[name]}")
    if name not in BUILDERS:
        raise ValueError(f"unknown benchmark {name!r}; have {sorted(BUILDERS)}")
    cache = Path(cache_dir) / f"{_cache_name(name, kw)}.json" if cache_dir else None
    if cache and cache.is_file() and not rebuild:
        es = EvalSet.load(cache)
        stale = _cache_is_stale(es, name, kw)
        if stale:
            log.error("[evalset] cached %s does NOT match the request (%s) -- "
                      "rebuilding. If you see this a lot, delete %s.",
                      name, stale, cache)
        else:
            log.info("[evalset] loaded cached %s: %s", name, es.stats())
            return es
    # absolute paths only: a manifest is meant to be handed to another process
    # (occlude_bench, a sweep script, a second machine) whose cwd is not ours,
    # and a relative "../Dataset/..." silently resolves to nothing there.
    es = BUILDERS[name](Path(data_root).expanduser().resolve(), **kw)
    _sanity(es)
    if cache:
        es.save(cache)
        _write_exclusion_list(es, cache)
    return es


def _write_exclusion_list(es: EvalSet, manifest: Path) -> Path | None:
    """For self-made holdouts, dump the gallery image names that training must drop.

    A held-out split carved from a training corpus is only meaningful if the
    training run actually stops seeing it. Emitting the list next to the manifest
    makes that a one-flag change (``train_dataset --exclude-images``) instead of
    something everyone has to remember.
    """
    if "heldout_ids" not in (es.meta or {}):
        return None
    p = manifest.with_suffix(".excluded_images.json")
    p.write_text(json.dumps({
        "benchmark": es.name,
        "why": "these images belong to held-out identities; training on them "
               "would leak the evaluation set",
        "image_ids": sorted({g["image_id"] for g in es.gallery}),
    }, indent=1), encoding="utf-8")
    log.warning("[evalset:%s] EXCLUDE THESE FROM TRAINING -> %s "
                "(pass it to train_dataset --exclude-images)", es.name, p)
    return p


def _sanity(es: EvalSet) -> None:
    """Loud warnings for the failure modes that silently corrupt a results table."""
    st = es.stats()
    if not es.gallery or not es.queries:
        raise RuntimeError(f"[evalset:{es.name}] empty gallery or query set")
    if st["queries_without_gallery_match"]:
        log.warning("[evalset:%s] %d query id(s) have NO gallery image -- those "
                    "queries are unanswerable and will drag every metric down",
                    es.name, st["queries_without_gallery_match"])
    exp = (es.meta or {}).get("expected")
    if exp:
        for k, v in exp.items():
            got = st.get(k)
            if got != v:
                log.warning("[evalset:%s] %s = %s but the published protocol "
                            "expects %s -- your numbers will NOT be comparable "
                            "to the literature until this matches",
                            es.name, k, got, v)
    log.info("[evalset:%s] %s", es.name, st)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", nargs="+", default=[], choices=sorted(BUILDERS))
    ap.add_argument("--describe", nargs="+", default=[], choices=sorted(BUILDERS))
    ap.add_argument("--data-root", default=str(
        Path(__file__).resolve().parents[2] / "Dataset"))
    ap.add_argument("--out", default=None, help="manifest cache dir")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--cuhk-json", default=None)
    ap.add_argument("--cuhk-images", default=None)
    ap.add_argument("--nomad-prompt-style", default="default",
                    choices=["default", "templated", "templated_neg",
                             "contrastive"])
    ap.add_argument("--nomad-split", default="val",
                    choices=["train", "val"],
                    help="which NOMAD actor split becomes the gallery; it is "
                         "part of the cache name, so switching it can no "
                         "longer reuse the other split's manifest")
    ap.add_argument("--heldout-ids", type=int, default=1000,
                    help="cuhk_heldout: how many identities to hold out")
    ap.add_argument("--heldout-seed", type=int, default=0,
                    help="cuhk_heldout: fixes WHICH identities -- changing it "
                         "invalidates every number already measured on the "
                         "previous split")
    ap.add_argument("--max-captions-per-image", type=int, default=2,
                    help="cuhk_heldout: the HF export has 4-8 augmented "
                         "captions per image; 2 matches the official scale")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cache = args.out or str(Path(args.data_root) / "Result" / "evalsets")
    report = {}
    for name in (args.build or args.describe):
        try:
            es = build(name, args.data_root,
                       cache_dir=None if args.describe and not args.build else cache,
                       rebuild=args.rebuild, cuhk_json=args.cuhk_json,
                       cuhk_images=args.cuhk_images,
                       nomad_prompt_style=args.nomad_prompt_style,
                       nomad_split=args.nomad_split,
                       heldout_ids=args.heldout_ids, seed=args.heldout_seed,
                       max_captions_per_image=args.max_captions_per_image)
            report[name] = es.stats()
        except (FileNotFoundError, RuntimeError, ImportError) as e:
            report[name] = {"ERROR": str(e).split("\n")[0]}
            print(f"\n### {name}: NOT AVAILABLE\n{e}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
