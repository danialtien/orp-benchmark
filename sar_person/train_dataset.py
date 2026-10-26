"""Fine-tune CLIP + LoRA on one or more text-to-image person datasets.

Training counterpart of ``eval_mals`` (eval-only). It loads (image, caption)
pairs from any mix of the project's caption-annotated datasets, holds out a small
retrieval-eval subset, fine-tunes the LoRA adapters with
``ClipEncoder.train_lora`` (symmetric InfoNCE + hard-negative margin), and writes
the checkpoint + before/after retrieval metrics.

Datasets (presets, resolved against --data-root, default ../Dataset):
    mals         MALS gene_crop + gene_attrs json          (128k, ground-level)
    cuhk         CUHK-PEDES parquet   (GONE: replaced by the author release)
    cuhk_author  CUHK-PEDES official images + official train split (PREFER)
    rstp         RSTPReid parquet                           (ground-level)
    icfg         ICFG-PEDES.json + imgs (unzip first)       (ground-level)
    forestperson ForestPerson forestperson_reid.json        (FOREST, domain)
    forestsynth  ForestSynth_v2 jsonl (per-person crops)    (DRONE/FOREST, domain)
    nomad        NOMAD/crop (scripts/prepare_nomad.py out)  (REAL DRONE, 100 IDs)

Two multi-dataset modes:
    sequential (default) : train on each dataset in the given order, the SAME
                           adapter carried forward — i.e. staged curriculum /
                           domain adaptation (pretrain -> fine-tune).
    --joint              : pool all datasets into one set and train once.

Why not ``sar_person.run``? ``run`` builds a full N×N similarity matrix +
heatmaps — fine for the ~20-row workbook, impossible at 10^4–10^6 images. Here
the model trains on the whole (sampled) set but is *evaluated* on a capped
subset, so metrics stay tractable.

Checkpoint format is the usual ``{"epoch": n, "trainable": {...}}`` and loads
unchanged in SAR_Pipeline (``--lora-ckpt``).

Examples
--------
# Single dataset (MALS) on the GPU box:
python -m sar_person.train_dataset --datasets mals --encoder clip \
    --out ../Dataset/Test_SAR_Person/mals_lora --epochs 20 --limit 40000

# Staged curriculum: pretrain on MALS, then adapt to the forest/drone sets:
python -m sar_person.train_dataset --datasets mals forestsynth forestperson \
    --encoder clip --out ../Dataset/Test_SAR_Person/staged --epochs 10

# Joint (pooled) training of the two forest sets:
python -m sar_person.train_dataset --datasets forestsynth forestperson --joint \
    --encoder clip --out ../Dataset/Test_SAR_Person/forest_joint --epochs 15

# CPU smoke test of the plumbing (no torch/weights):
python -m sar_person.train_dataset --datasets forestperson --encoder mock \
    --limit 200 --eval-size 30 --out /tmp/smoke

# NOMAD alone (real drone imagery, actor-level 80/20 split comes from the data):
python -m sar_person.train_dataset --datasets nomad --encoder clip \
    --out ../Dataset/Test_SAR_Person/nomad_lora --epochs 15 --limit 20000

# The intended curriculum: ground-level pretrain -> real aerial adaptation.
python -m sar_person.train_dataset --datasets mals nomad --encoder clip \
    --out ../Dataset/Test_SAR_Person/mals_to_nomad --epochs 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import metrics_viz as mv

log = logging.getLogger("train_dataset")

# --------------------------------------------------------------------------- #
# dataset presets. Paths resolve against --data-root (default ../Dataset).     #
#   kind: "mals" (gene_attrs) | "parquet" | "json"                            #
# --------------------------------------------------------------------------- #
PRESETS = {
    "mals": {"kind": "mals", "attrs": "MALS/gene_attrs/g_4x_attrs.json"},
    # HuggingFace parquet mirror. Same 11 003 identities as the official train
    # split, but the captions were augmented to 4-8 per image by an unknown
    # generator (official: 2). More supervision than IRRA/RDE saw, from a
    # different distribution -- fine for domain adaptation, wrong for a
    # like-for-like comparison. Prefer `cuhk_official` below.
    "cuhk": {"kind": "parquet", "parquet": "CUHK-PEDES/data",
             "images": "CUHK-PEDES/images"},
    # The official train split reconstructed from reid_raw.json (sha256 recorded
    # in provenance.json): 34 054 images, 11 003 ids, **2 captions per image**.
    # This is the supervision IRRA, RDE and APTM trained on, so it is the one to
    # use when the claim is "same scale, fewer parameters".
    "cuhk_official": {"kind": "json",
                      "json": "CUHK-PEDES-train/annotations/train.jsonl",
                      "image_root": "CUHK-PEDES-train",
                      "split": ("split", "train")},
    # The same official train split (34 054 images, 11 003 ids, 2 captions each)
    # but on the AUTHOR'S OWN IMAGES, obtained on request under the CUHK terms.
    # `cuhk_official` above reads pixels from unofficial HF mirrors: fine for a
    # sweep, not fine for a published table. Train the reported checkpoint on
    # this one. Built by `python scripts/prepare_cuhk.py`; see
    # Dataset/CUHK-PEDES/cuhk_provenance.json for what is official and what is
    # reconstructed.
    "cuhk_author": {"kind": "json",
                    "json": "CUHK-PEDES/annotations/train.jsonl",
                    "image_root": "CUHK-PEDES",
                    "split": ("split", "train")},
    # The SAME images and the SAME official split as `cuhk_author`, with the
    # captions regenerated from the pixels by a VLM under the two-tier contract
    # in `sar_person/reid_vlm.py` (built by `scripts/reid_vlm_captions.py`;
    # provenance + the leak guards' results in annotations/vlm/manifest.json).
    # `train.jsonl` is never overwritten, so arm D0 stays reproducible while
    # these train.
    #   _vlm  1 caption  per image (VLM only)      -> arm D3
    #   _aug  3 captions per image (2 human + VLM) -> arm D1
    # NOTE the pair counts: `load_json_pairs` explodes every caption into its
    # own pair, so _aug is ~102k pairs where cuhk_author is 68 126. Any --limit
    # carried over from a cuhk_author command is wrong for _aug -- that
    # confusion is exactly what arm D2 (volume-matched) exists to rule out.
    "cuhk_author_vlm": {"kind": "json",
                        "json": "CUHK-PEDES/annotations/vlm/train_vlm.jsonl",
                        "image_root": "CUHK-PEDES",
                        "split": ("split", "train")},
    "cuhk_author_aug": {"kind": "json",
                        "json": "CUHK-PEDES/annotations/vlm/train_aug.jsonl",
                        "image_root": "CUHK-PEDES",
                        "split": ("split", "train")},
    "rstp": {"kind": "parquet", "parquet": "RSTPReid/data",
             "images": "RSTPReid/images", "split": ("split", "train")},
    "icfg": {"kind": "json",
             "json": "ICFG-PEDES/ICFG-PEDES/ICFG-PEDES.json",
             "image_root": "ICFG-PEDES/ICFG-PEDES/imgs",
             "split": ("split", "train")},
    "forestperson": {"kind": "json",
                     "json": "ForestPerson/forestperson_reid.json",
                     "image_root": ".", "split": ("split", "train")},
    "forestsynth": {"kind": "json",
                    "json": "GenImage/ForestSynth_v2/annotations.jsonl",
                    "image_root": "GenImage/ForestSynth_v2/images",
                    "crop_out": "GenImage/ForestSynth_v2/_person_crops"},
    # NOMAD (WACV 2024) aerial person crops, prepared by
    # scripts/prepare_nomad.py. Actor-level 80/20 split and one caption per
    # actor -> the loader tags pairs with group/eval_group/split (see
    # nomad_data.py) and this module honours all three.
    "nomad": {"kind": "nomad", "crop_root": "NOMAD/crop"},
}
DATASET_NAMES = list(PRESETS)

# --------------------------------------------------------------------------- #
# Fixed external benchmarks (--eval-on). Every run's own eval split differs in  #
# size and difficulty -- NOMAD scores on a 20x20 matrix, CUHK on up to          #
# 1000x1000 -- so `final_R@1` CANNOT be compared across dataset groups. A       #
# benchmark here is a single held-out set every checkpoint is scored on, which  #
# is the only honest way to rank runs that trained on different data.           #
#                                                                              #
# Each entry MUST be disjoint from what the matching training preset consumes.  #
# rstp trains on split=train; the benchmark reads split=test -> no leakage even #
# when RSTPReid is part of the training mix.                                    #
# --------------------------------------------------------------------------- #
BENCHMARKS = {
    "rstp": {"kind": "parquet", "parquet": "RSTPReid/data",
             "images": "RSTPReid/images", "split": ("split", "test"),
             "note": "RSTPReid test split (1000 identities), disjoint from the "
                     "rstp training preset (split=train)"},
    # DISABLED. The CUHK-PEDES parquet shards in this project are train-only and
    # carry no `split` column, so the filter below was a silent no-op and this
    # "benchmark" scored the TRAINING data. Re-enable only after
    # Dataset/CUHK-PEDES/reid_raw.json is in place (see eval_sets._CUHK_HINT),
    # and prefer `python -m sar_person.eval_protocol --benchmark cuhk` anyway.
    "cuhk": {"kind": "parquet", "parquet": "CUHK-PEDES/data",
             "images": "CUHK-PEDES/images", "split": ("split", "test"),
             "disabled": "the CUHK-PEDES parquet has no test split -- this "
                         "benchmark was evaluating on training data",
             "note": "CUHK-PEDES test split"},
    "nomad": {"kind": "nomad", "crop_root": "NOMAD/crop",
              "note": "NOMAD's 20 held-out actors (actor-disjoint from train)"},
}


def load_benchmark_pairs(name: str, args) -> list[dict]:
    """Load the fixed evaluation set named by ``--eval-on``.

    Returns one pair PER DISTINCT IMAGE (or per identity for NOMAD), because a
    retrieval matrix needs exactly one correct column per row -- duplicated
    captions would create ties and silently deflate R@1.
    """
    root = Path(args.data_root)
    spec = BENCHMARKS[name]
    if spec.get("disabled"):
        raise SystemExit(
            f"--eval-on {name} is disabled: {spec['disabled']}.\n"
            f"Use `python -m sar_person.eval_protocol --benchmark {name}` for a "
            f"full-gallery, identity-aware evaluation instead.")
    if spec["kind"] == "parquet":
        from .parquet_data import load_parquet_pairs
        pairs = load_parquet_pairs(
            str(root / spec["parquet"]), str(root / spec["images"]),
            limit=None, seed=args.seed, split_col_filter=spec["split"],
            max_captions_per_image=1)
    elif spec["kind"] == "nomad":
        from .nomad_data import load_nomad_pairs
        pairs = [p for p in load_nomad_pairs(str(root / spec["crop_root"]),
                                             prompt_style=args.nomad_prompt_style,
                                             limit=None, seed=args.seed)
                 if p.get("split") == "val"]
    else:
        raise ValueError(f"unknown benchmark kind for {name!r}")

    pairs = _dedup_by_image(pairs, "eval_group" if any(
        p.get("eval_group") for p in pairs) else None)
    rng = np.random.default_rng(args.seed)
    if args.eval_on_size and args.eval_on_size < len(pairs):
        idx = rng.choice(len(pairs), args.eval_on_size, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
    log.info("[benchmark] %s: %d pair(s) — %s", name, len(pairs), spec["note"])
    return pairs


_EXCLUDE_CACHE: dict = {}


def _excluded_image_names(args) -> set:
    """Image names that must never reach training, from --exclude-images.

    Needed because ``eval_sets --build cuhk_heldout`` carves its evaluation set
    out of the *training* corpus: without this the same identities sit on both
    sides of the split and the resulting numbers are meaningless.
    """
    paths = getattr(args, "exclude_images", None) or []
    key = tuple(paths)
    if key in _EXCLUDE_CACHE:
        return _EXCLUDE_CACHE[key]
    names: set = set()
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        got = set(d["image_ids"] if isinstance(d, dict) else d)
        log.info("[exclude] %d held-out image name(s) from %s", len(got), p)
        names |= got
    if names:
        log.warning("[exclude] %d image(s) will be withheld from ALL training "
                    "datasets in this run", len(names))
    _EXCLUDE_CACHE[key] = names
    return names


def load_dataset_pairs(name: str, args, limit: int | None) -> list[dict]:
    """Return {image, prompt, image_id} pairs, and say so when --limit bit.

    The default ``--limit 20000`` has silently truncated a run twice: once
    cutting 40% of ICFG, once dropping the identity count from 3102 to 2480 (and
    with it the ID classifier's output dimension). A loader returning exactly
    ``limit`` pairs is proof the cap was reached, so warn there -- the only false
    positive is a dataset whose size happens to equal the limit exactly.
    """
    pairs = _load_dataset_pairs(name, args, limit)
    if limit is not None and len(pairs) >= limit:
        log.warning(
            "[limit] %s returned exactly %d pair(s) == --limit. The dataset is "
            "almost certainly LARGER and was truncated; identity count and "
            "num_classes are truncated with it. Pass the full size explicitly "
            "(icfg: 34674, cuhk_author: 68126, cuhk_author_vlm: 34054, "
            "cuhk_author_aug: ~102180 -- the exact number is in "
            "annotations/vlm/manifest.json) or --limit 0 for everything.",
            name, len(pairs))
    return pairs


def _load_dataset_pairs(name: str, args, limit: int | None) -> list[dict]:
    """Return {image, prompt, image_id} pairs for one preset dataset."""
    root = Path(args.data_root)
    preset = PRESETS[name]
    kind = preset["kind"]

    if kind == "mals":
        from .mals import load_mals_pairs
        return load_mals_pairs(str(root / preset["attrs"]), None, limit=limit,
                               seed=args.seed, stride=args.stride)
    if kind == "parquet":
        from .parquet_data import load_parquet_pairs
        return load_parquet_pairs(
            str(root / preset["parquet"]),
            str(root / preset["images"]), limit=limit, seed=args.seed,
            split_col_filter=preset.get("split"),
            max_captions_per_image=args.max_captions,
            exclude_names=_excluded_image_names(args))
    if kind == "nomad":
        from .nomad_data import load_nomad_pairs
        return load_nomad_pairs(
            str(root / preset["crop_root"]),
            prompt_style=args.nomad_prompt_style, limit=limit, seed=args.seed,
            altitudes=args.nomad_altitudes, min_visibility=args.nomad_min_vis,
            activities=args.nomad_activities)
    if kind == "json":
        from .json_data import load_json_pairs
        image_root = root if preset["image_root"] == "." else \
            root / preset["image_root"]
        crop_out = str(root / preset["crop_out"]) if "crop_out" in preset else None
        return load_json_pairs(str(root / preset["json"]), str(image_root),
                               limit=limit, seed=args.seed,
                               split_filter=preset.get("split"),
                               crop_out=crop_out)
    raise ValueError(f"unknown kind {kind!r} for dataset {name!r}")


def _dedup_by_image(pairs: list[dict], key_field: str | None = None) -> list[dict]:
    """Keep the first pair per key. ``key_field`` (e.g. "eval_group") lets a
    loader collapse many samples that share one caption down to one."""
    seen, out = set(), []
    for p in pairs:
        key = (p.get(key_field) if key_field else None) or \
            p.get("image_id") or p["image"]
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _encode(fn, items, batch):
    try:
        return fn(items, batch=batch)
    except TypeError:
        return fn(items)


def _split_train_eval(pairs, eval_size, rng, eval_ids=300):
    """Hold out a clean retrieval eval; rest train.

    ``eval_ids`` (default 300) holds out that many IDENTITIES and keeps ALL of
    their images in the eval pool. That fixes two faults of the old image-level
    path, both of which cost real accuracy on ICFG-PEDES:

    * ICFG records carry no ``image_id``, so ``_get`` fell back to ``id`` -- the
      PERSON id. ``_dedup_by_image`` therefore collapsed 34 674 images to 3 102
      rows, ``eval_n`` capped at ``3102 // 5 = 620``, and every image of those
      620 identities left training: **2 483 of 3 102 identities survived, 79% of
      the data**, while ``--eval-size 3000`` silently became 620.
    * One image per identity makes mINP degenerate (a single positive is always
      the last one), so the identity-level ``t2i_id_*`` metrics never fired and
      checkpoint selection could not see the tail of the ranking.

    Holding out identities keeps the split identity-disjoint -- no leakage, same
    as before -- but pays 300 identities instead of 620, and the eval pool then
    holds ~11 images per identity, which is what mINP needs.

    ``eval_ids=0`` restores the old image-level behaviour for reproducing runs
    made before 2026-09-02.

    Two refinements, both driven by optional keys a loader may attach — datasets
    that don't set them get exactly the previous behaviour:

    ``split``      pairs tagged ``"val"`` become the eval pool and never train.
                   Needed when the split is over IDENTITIES rather than images
                   (NOMAD: 80/20 by actor); a random image-level holdout would
                   otherwise put the same caption on both sides of the split.
    ``eval_group`` key the eval pool is deduplicated on, instead of image_id.
                   NOMAD again: ~90 crops per actor share one caption, so
                   without grouping the eval matrix has ~90 tied "correct"
                   columns per row and R@1 stops measuring retrieval quality.
    """
    rng.shuffle(pairs)
    grouped = any(p.get("eval_group") for p in pairs)
    key_field = "eval_group" if grouped else None

    val_pairs = [p for p in pairs if p.get("split") == "val"]
    if val_pairs:
        train_pairs = [p for p in pairs if p.get("split") != "val"]
        eval_pool = _dedup_by_image(val_pairs, key_field)
        eval_pairs = eval_pool[:max(2, min(eval_size, len(eval_pool)))]
        log.info("[split] dataset-provided: %d train pair(s), %d eval pair(s) "
                 "from %d held-out group(s)", len(train_pairs), len(eval_pairs),
                 len(eval_pool))
        return train_pairs, eval_pairs

    # --- identity holdout (default) ------------------------------------- #
    groups = [p.get("group") for p in pairs]
    if eval_ids and not grouped and all(g is not None for g in groups):
        uniq = list(dict.fromkeys(groups))          # pairs are already shuffled
        n_hold = max(2, min(int(eval_ids), max(2, len(uniq) // 5)))
        held = set(uniq[:n_hold])
        train_pairs = [p for p in pairs if p.get("group") not in held]
        # dedup on the IMAGE PATH, not image_id: on ICFG image_id is the person
        # id, and deduping on it would collapse the pool back to one row each.
        pool = _dedup_by_image([p for p in pairs if p.get("group") in held],
                               "image")
        eval_pairs = pool[:max(2, min(eval_size, len(pool)))]
        log.info("[split] identity holdout: %d/%d identity/identities held out "
                 "(%d train pair(s) over %d identity/identities), eval = %d "
                 "image(s) from %d pooled, %.1f image(s) per identity",
                 n_hold, len(uniq), len(train_pairs),
                 len(uniq) - n_hold, len(eval_pairs), len(pool),
                 len(eval_pairs) / max(len({p.get("group")
                                            for p in eval_pairs}), 1))
        return train_pairs, eval_pairs

    if eval_ids:
        log.warning("[split] --eval-ids %d requested but the loader gave no "
                    "usable 'group' field; falling back to the image-level "
                    "holdout, which consumes up to 20%% of identities",
                    eval_ids)
    eval_pool = _dedup_by_image(pairs, key_field)
    eval_n = min(eval_size, max(2, len(eval_pool) // 5))
    eval_pairs = eval_pool[:eval_n]
    eval_ids = {p.get("image_id") or p["image"] for p in eval_pairs}
    if grouped:      # also pull the rest of a held-out group out of training
        held = {p.get("eval_group") for p in eval_pairs}
        train_pairs = [p for p in pairs if p.get("eval_group") not in held]
    else:
        train_pairs = [p for p in pairs
                       if (p.get("image_id") or p["image"]) not in eval_ids]
    return train_pairs, eval_pairs


def _retrieval_metrics(enc, eval_pairs, batch):
    images = [p["image"] for p in eval_pairs]
    prompts = [p["prompt"] for p in eval_pairs]
    n = len(eval_pairs)
    gt_col = list(range(n))
    S = mv.similarity(_encode(enc.encode_images, images, batch),
                      _encode(enc.encode_texts, prompts, batch))
    m = mv.compute_metrics(S, gt_col)
    order = np.argsort(-S, axis=1)
    gt = np.asarray(gt_col)
    m["i2t_R@10"] = float(np.mean([gt[i] in order[i, :min(10, n)]
                                   for i in range(n)]))

    # Identity-level text->image metrics, so --select-metric can optimise the
    # END of the ranking. mv.compute_metrics assumes ONE correct column per row,
    # which makes mINP undefined there; rank_metrics takes identity labels and
    # scores every image of the person as a positive -- the same definition
    # eval_protocol uses on the real gallery. Only emitted when the eval pool
    # actually holds several images per identity, otherwise mINP is a constant 1.
    groups = [p.get("group") for p in eval_pairs]
    if all(g is not None for g in groups) and len(set(groups)) < n:
        from .eval_protocol import rank_metrics
        pid = np.asarray([str(g) for g in groups])
        idm = rank_metrics(np.ascontiguousarray(S.T, dtype=np.float64),
                           pid, pid, warn_skipped=False)
        for k, v in idm.items():
            if isinstance(v, (int, float)):
                m[f"t2i_id_{k}"] = float(v)
    return m, S


def _train_once(enc, train_pairs, eval_pairs, args, ckpt, trial_seed,
                on_epoch=None, on_eval=None, info=None):
    """One train_lora pass (clip) or a simulate_finetune (mock); returns
    (metrics_after, S_after).

    When ``--eval-every`` is on, the eval split below is scored DURING training
    and the best-scoring epoch is what ends up in ``ckpt`` and in the live
    encoder -- so the metrics returned here, the adapter the next curriculum
    stage continues from, and the file you deploy are all the same epoch."""
    if args.encoder == "clip":
        # "group" is forwarded so train_lora can build identity-disjoint batches
        # for datasets where many samples share one caption (NOMAD). Absent for
        # every other preset -> train_lora keeps its plain random permutation.
        lora_pairs = [{"img_path": p["image"], "prompt": p["prompt"],
                       "group": p.get("group")}
                      for p in train_pairs]

        mid_eval = None
        if args.eval_every and args.eval_every > 0:
            def mid_eval(ep):
                m, _ = _retrieval_metrics(enc, eval_pairs, args.enc_batch)
                return m

        sel = enc.train_lora(lora_pairs, epochs=args.epochs, lr=args.lr,
                       lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                       seed=trial_seed,
                       batch=args.batch, amp=not args.no_amp,
                       ckpt_path=str(ckpt),
                       hard_neg_weight=args.hard_neg_weight, margin=args.margin,
                       deterministic=args.deterministic, on_epoch=on_epoch,
                       occ_prob=args.occ_prob, occ_variants=args.occ_variants,
                       occ_levels=args.occ_levels, oil_root=args.oil_root,
                       loss=args.loss, num_instance=args.num_instance,
                       temp=args.temp,
                       sdm_temp=args.sdm_temp, sdm_weight=args.sdm_weight,
                       sampler=args.sampler,
                       lr_factor=args.lr_factor,
                       bias_lr_factor=args.bias_lr_factor,
                       weight_decay=args.weight_decay,
                       weight_decay_bias=args.weight_decay_bias,
                       optimizer=args.optimizer,
                       lora_attn=args.lora_attn,
                       lr_schedule=args.lr_schedule,
                       warmup_epochs=args.warmup_epochs,
                       eval_fn=mid_eval, eval_every=args.eval_every,
                       on_eval=on_eval, select_metric=args.select_metric,
                       full_finetune=args.full_finetune,
                       grad_clip=args.grad_clip,
                       id_loss_weight=args.id_loss_weight,
                       mlm_loss_weight=args.mlm_loss_weight,
                       cmt_depth=args.cmt_depth,
                       train_aug=args.aug)
        if info is not None and isinstance(sel, dict):
            info.update(sel)
        return _retrieval_metrics(enc, eval_pairs, args.enc_batch)
    # mock: simulate the effect of fine-tuning on the eval set (plumbing only)
    imgs = _encode(enc.encode_images, [p["image"] for p in eval_pairs],
                   args.enc_batch)
    txts = _encode(enc.encode_texts, [p["prompt"] for p in eval_pairs],
                   args.enc_batch)
    gt = list(range(len(eval_pairs)))
    S = mv.similarity(imgs, enc.simulate_finetune(imgs, txts, gt))
    return mv.compute_metrics(S, gt), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mals"],
                    choices=DATASET_NAMES,
                    help="one or more datasets (order matters for staged mode)")
    ap.add_argument("--joint", action="store_true",
                    help="pool all datasets and train once (default: staged, "
                         "carrying the adapter dataset-to-dataset)")
    ap.add_argument("--data-root", default=str(Path(__file__).resolve()
                                               .parents[2] / "Dataset"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into an --out that already holds lora_ckpt.pt. "
                         "Default: STOP, so an existing checkpoint is never lost.")
    ap.add_argument("--max-captions", type=int, default=None)
    # training
    ap.add_argument("--encoder", choices=["clip", "mock"], default="clip")
    ap.add_argument("--pretrain", default="ViT-B/16")
    ap.add_argument("--img-size", nargs=2, type=int, default=[224, 224],
                    metavar=("H", "W"),
                    help="input resolution. 224 224 (default) letterboxes, "
                         "which suits aerial crops of varying aspect. Use "
                         "384 128 -- plain resize, no padding -- to match "
                         "IRRA/RDE/TBPS-CLIP; person crops are already ~3:1, so "
                         "a square frame spends most of itself on grey bars. "
                         "The eval run MUST use the same value")
    ap.add_argument("--gpu", default="1",
                    help="physical GPU id to use, e.g. '1' (default) or '0'; "
                         "sets CUDA_VISIBLE_DEVICES so torch still sees it as "
                         "cuda:0 and AMP stays on. 'all' = don't restrict. An "
                         "existing CUDA_VISIBLE_DEVICES in the environment wins.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("-t", "--trials", type=int, default=1,
                    help="retry rounds per dataset (best R@1 kept)")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", default="16", metavar="A|rank",
                    help="LoRA scaling numerator; the adapter contributes "
                         "(alpha/rank) * B@A. Default '16' reproduces every run "
                         "made before this flag existed. Pass 'rank' to set "
                         "alpha = --lora-rank, which pins the scaling at 1.0 — "
                         "REQUIRED for an honest rank sweep, because with a "
                         "fixed alpha=16 sweeping rank 4->64 also swings the "
                         "update magnitude 4.0->0.25 and you end up measuring "
                         "scaling, not capacity.")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-schedule", default="cosine",
                    choices=["cosine", "constant"],
                    help="'cosine' (default): linear warmup for "
                         "--warmup-epochs, then cosine decay to 0, the "
                         "schedule IRRA/TBPS-CLIP/RDE use. 'constant' holds "
                         "--lr flat for every epoch -- the behaviour before "
                         "2026-08-13, kept only to reproduce older runs. Held "
                         "flat, 5e-4 never settles: the adapter keeps taking "
                         "full-size steps around the minimum and the final "
                         "epoch is a draw from that oscillation")
    ap.add_argument("--warmup-epochs", type=int, default=5, metavar="E",
                    help="linear LR warmup length for --lr-schedule cosine "
                         "(default 5; ignored when the schedule is constant). "
                         "Clamped to epochs-1 so the LR always gets to decay")
    ap.add_argument("--eval-every", type=int, default=5, metavar="N",
                    help="score the held-out eval split every N epochs DURING "
                         "training and keep the best-scoring epoch (by "
                         "--select-metric) as the checkpoint. 0 disables "
                         "selection, which restores the old behaviour: the "
                         "saved adapter is the LAST epoch, best only if the "
                         "run never overfits. The last epoch is always kept "
                         "alongside as <name>_last.pt")
    ap.add_argument("--select-metric", default="i2t_R@1", metavar="KEY",
                    choices=["i2t_R@1", "i2t_R@5", "i2t_MRR", "t2i_R@1",
                             "pos_neg_gap"],
                    help="metric maximised when picking the best epoch "
                         "(default i2t_R@1). i2t_MRR is the steadier signal on "
                         "a small eval split -- R@1 moves in whole samples, so "
                         "on --eval-size 200 one flipped pair is 0.5 points "
                         "and epoch choice starts tracking noise. NOTE: mINP "
                         "is deliberately absent; metrics_viz.compute_metrics "
                         "assumes one correct column per row and cannot "
                         "compute it (see sar_person.eval_protocol for "
                         "identity-level metrics)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--enc-batch", type=int, default=64)
    ap.add_argument("--no-amp", action="store_true")
    # --- objective -------------------------------------------------------- #
    ap.add_argument("--loss", default="infonce",
                    choices=["infonce", "sdm", "sdm+infonce"],
                    help="'infonce' treats every (image, caption) pair as the "
                         "only positive in its batch -- false on ICFG, where an "
                         "identity has ~11 crops that one caption describes. "
                         "'sdm' (Jiang & Ye, CVPR'23) matches the similarity "
                         "distribution to the identity-match distribution and "
                         "needs identity labels + PK sampling, both handled "
                         "automatically. Keep 'infonce' runs as the ablation")
    ap.add_argument("--num-instance", type=int, default=4, metavar="K",
                    help="images per identity in a batch (SDM only). The batch "
                         "then holds batch//K identities. IRRA uses 4")
    ap.add_argument("--sdm-temp", type=float, default=0.02,
                    help="SDM temperature (IRRA's default)")
    ap.add_argument("--sdm-weight", type=float, default=1.0)
    ap.add_argument("--lora-attn", action="store_true",
                    help="also put a parallel LoRA branch on every attention "
                         "block (default: MLP only). Untested at scale -- run "
                         "the SDM change on its own first, or you will not know "
                         "which of the two moved the number")
    ap.add_argument("--full-finetune", action="store_true",
                    help="full backbone fine-tuning: unfreeze ALL CLIP params "
                         "instead of injecting LoRA adapters. ~149 M params "
                         "for ViT-B/16 vs 3 M for LoRA r=16. Use a much "
                         "smaller learning rate (--lr 1e-5 to 5e-5) and "
                         "fewer epochs (10-30) to avoid catastrophic forgetting. "
                         "Checkpoint will contain the FULL model state (~600 MB "
                         "fp32), not a small adapter.")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    metavar="NORM",
                    help="gradient norm clipping (default 0 = disabled for LoRA "
                         "runs). Automatically set to 1.0 when --full-finetune "
                         "is passed unless you override it explicitly. Clipping "
                         "prevents gradient explosions that are common in full "
                         "backbone fine-tuning.")
    ap.add_argument("--hard-neg-weight", type=float, default=0.5,
                    help="ignored under --loss sdm: the margin pushes apart the "
                         "hardest off-diagonal pair, which under PK sampling is "
                         "usually two crops of the same person")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--deterministic", action="store_true")
    # --- IRRA auxiliary objectives (train-time only) ---------------------- #
    # IRRA's default is loss_names='sdm+id+mlm'. This repo shipped only the sdm
    # (or infonce) term, so a run here was never a like-for-like baseline. Both
    # default to 0.0: with the flags left alone, training is bit-identical to
    # what it was before these heads existed.
    ap.add_argument("--id-loss-weight", type=float, default=0.0, metavar="W",
                    help="weight of the identity classification loss (IRRA "
                         "parity: 1.0). Needs identity labels in the data. The "
                         "classifier is discarded after training, so it does "
                         "NOT change the parameter count at inference -- but "
                         "report it in the train-time column.")
    ap.add_argument("--mlm-loss-weight", type=float, default=0.0, metavar="W",
                    help="weight of the IRR / masked-language-modelling loss "
                         "(IRRA parity: 1.0). Adds ~39.3M TRAIN-ONLY parameters "
                         "and a second forward pass through the text tower; "
                         "expect +35-50%% time per epoch. Turn --id-loss-weight "
                         "on first and confirm it helps before adding this.")
    ap.add_argument("--sampler", default="auto",
                    choices=["auto", "pk", "disjoint", "random"],
                    help="batch sampler. 'auto' (default) keeps the historical "
                         "coupling: PK for --loss sdm, identity-disjoint for "
                         "everything else. Pin it to 'pk' or 'disjoint' when "
                         "comparing SDM against InfoNCE -- under 'auto' that "
                         "comparison changes the objective AND the sampler at "
                         "the same time, and the delta cannot be attributed. "
                         "'random' is a plain shuffle with no identity "
                         "constraint -- this is what IRRA's published configs "
                         "use (sampler: random), so it is the one to pass for "
                         "a parity run. 'disjoint' is NOT the same thing: it "
                         "forbids a repeated identity rather than making one "
                         "rare.")
    # ---- IRRA optimizer parity (added 2026-09-15) ----------------------- #
    ap.add_argument("--lr-factor", type=float, default=1.0, metavar="F",
                    help="LR multiplier for the randomly-initialised auxiliary "
                         "heads (ID classifier, cross-modal transformer, MLM "
                         "head). IRRA parity: 5.0 (configs.yaml lr_factor). "
                         "Default 1.0 reproduces every run made before "
                         "2026-09-15, where all parameters shared one LR and "
                         "the 44.8M random-init aux parameters crawled at the "
                         "backbone's rate -- the leading suspect for rounds B "
                         "and C gaining only +0.59 / +0.47 R@1 from ID+MLM.")
    ap.add_argument("--bias-lr-factor", type=float, default=1.0, metavar="F",
                    help="LR multiplier for bias tensors, and their weight "
                         "decay is forced to 0. IRRA parity: 2.0.")
    ap.add_argument("--weight-decay", type=float, default=1e-4, metavar="W",
                    help="weight decay (default 1e-4, historical). IRRA "
                         "parity: 4e-5.")
    ap.add_argument("--weight-decay-bias", type=float, default=None, metavar="W",
                    help="separate weight decay for bias tensors. Unset "
                         "(default) means biases decay like everything else, "
                         "which is what every run before 2026-09-15 did. IRRA "
                         "parity: 0.")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"],
                    help="default 'adamw' (historical). IRRA uses 'adam' with "
                         "eps 1e-3, which is not a typo in their repo and "
                         "changes the first epochs at lr 1e-5.")
    ap.add_argument("--temp", type=float, default=0.07, metavar="T",
                    help="InfoNCE temperature (default 0.07, CLIP's own). "
                         "SDM uses --sdm-temp instead; the two are separate "
                         "knobs and only one is active per objective.")
    ap.add_argument("--aug", action="store_true",
                    help="train-time image augmentation: horizontal flip 0.5, "
                         "pad-10 + random crop, random erasing 0.5 -- the three "
                         "IRRA uses. Applied ONLY on the training path; the "
                         "gallery is never augmented. Off by default so older "
                         "runs stay reproducible, but every published baseline "
                         "has it: this is part of why 55.06 sits BELOW IRRA's "
                         "own 56.74 baseline row.")
    ap.add_argument("--cmt-depth", type=int, default=4, metavar="L",
                    help="cross-modal transformer depth for the MLM head. Leave "
                         "at 4 for the IRRA-parity row.")
    # --- occlusion-aware training (Table B: the train-occ row) ------------- #
    ap.add_argument("--occ-prob", type=float, default=0.0, metavar="P",
                    help="probability of occluding each training image "
                         "on the fly. 0 = the train-clean row (generalisation "
                         "to unseen occlusion); 0.5 = the train-occ row, the "
                         "setting MGCC uses. Report BOTH -- the delta is the "
                         "occlusion-aware ablation")
    ap.add_argument("--occ-variants", nargs="+",
                    default=["torso", "cutout", "foliage"],
                    help="occlusion types sampled during training. Add 'ogor' "
                         "to train against MGCC's instance library (needs "
                         "--oil-root)")
    ap.add_argument("--occ-levels", nargs="+", type=float,
                    default=[0.25, 0.5, 0.7],
                    help="coverage levels sampled during training")
    ap.add_argument("--oil-root", default=str(
        Path(__file__).resolve().parents[2] / "Dataset" / "OIL"),
        help="occlusion instance library, only used when 'ogor' is in "
             "--occ-variants (scripts/fetch_oil.py)")
    ap.add_argument("--init-ckpt", default=None,
                    help="resume from a previously trained adapter "
                         "(e.g. a prior run's lora_ckpt.pt). Injects LoRA and "
                         "loads its weights BEFORE training, so separate commands "
                         "can be chained instead of starting from zero-shot CLIP.")
    # data sizing
    ap.add_argument("--limit", type=int, default=20000,
                    help="max pairs PER dataset (train+eval). Use 0 (or any "
                         "value <= 0) for NO limit -- required to train on a "
                         "full split: the 20000 default silently drops 40%% of "
                         "ICFG and 70%% of CUHK, which is fine for a sweep and "
                         "wrong for a number you put next to IRRA")
    ap.add_argument("--eval-size", type=int, default=1000)
    ap.add_argument("--eval-ids", type=int, default=300, metavar="N",
                    help="hold out N IDENTITIES (with all their images) for the "
                         "mid-training eval, instead of a flat image holdout. "
                         "Keeps the split identity-disjoint while returning ~10%% "
                         "of the identities the old path consumed, and gives the "
                         "eval pool several images per identity so mINP is "
                         "defined. Capped at 20%% of identities. 0 = the "
                         "pre-2026-09-02 behaviour; runs are NOT comparable "
                         "across the two.")
    ap.add_argument("--eval-on", default=None, choices=list(BENCHMARKS),
                    help="ALSO score the trained adapter on a fixed external "
                         "benchmark, before and after training. This is what "
                         "makes runs on DIFFERENT dataset mixes comparable — "
                         "each run's own eval split has its own size and "
                         "difficulty, so `final_R@1` alone cannot rank them.")
    ap.add_argument("--eval-on-size", type=int, default=1000, metavar="N",
                    help="cap the --eval-on benchmark at N images (default "
                         "1000). Keep it IDENTICAL across every run you intend "
                         "to compare: R@1 depends on gallery size.")
    ap.add_argument("--exclude-images", nargs="+", default=None,
                    metavar="JSON",
                    help="one or more *.excluded_images.json written by "
                         "`eval_sets --build cuhk_heldout`. Required whenever "
                         "the evaluation set was carved out of a training "
                         "corpus, otherwise the holdout leaks into training.")
    ap.add_argument("--stride", type=int, default=1, help="MALS record thinning")
    ap.add_argument("--seed", type=int, default=0)
    # NOMAD-only knobs (ignored by every other preset). All three prompt
    # flavours are stored per crop by scripts/prepare_nomad.py, so switching
    # style or narrowing the altitude/visibility/activity slice needs no
    # re-cropping.
    ap.add_argument("--nomad-prompt-style", default="default",
                    choices=["default", "templated", "templated_neg",
                             "contrastive"],
                    help="NOMAD caption flavour: DEFAULT concatenation, APTM "
                         "attribute templates, or templates + negatives")
    ap.add_argument("--nomad-altitudes", nargs="+", type=int, default=None,
                    metavar="M",
                    help="restrict NOMAD to these flight altitudes in metres "
                         "(e.g. --nomad-altitudes 10). Default: whatever was "
                         "cropped.")
    ap.add_argument("--nomad-min-vis", type=int, default=0, metavar="PCT",
                    help="tighten NOMAD's visibility floor beyond the "
                         "prepare-time filter (10..100)")
    ap.add_argument("--nomad-activities", nargs="+", default=None,
                    metavar="LABEL",
                    help="keep only these NOMAD activity labels, e.g. Walking "
                         "Laying 'Hiding (Laying)'")
    # NOMAD actor-level check, run before AND after training (see
    # sar_person.eval_nomad_actors). This is the 100-identity replacement for
    # the 20-row Excel heatmap, and unlike `heatmap_after.png` -- which shows
    # this run's own eval subset, a different set for every dataset mix -- it
    # is the same 100 actors and the same 100 profile prompts in every run, so
    # the pictures are comparable across runs.
    ap.add_argument("--eval-nomad", default="auto",
                    choices=["auto", "all", "val", "off"],
                    help="score the adapter on NOMAD actor prototypes x profile "
                         "prompts and write heatmap_{before,after}_nomad.png. "
                         "'auto' (default) runs it whenever the NOMAD crop set "
                         "exists and picks the actor set for you: 'val' (the "
                         "20 held-out actors) if nomad is in --datasets, else "
                         "'all' 100. Forcing 'all' on a run that TRAINED on "
                         "nomad reports memorisation, not retrieval.")
    ap.add_argument("--eval-nomad-crops", type=int, default=20, metavar="N",
                    help="crops per actor for the prototype, spread evenly over "
                         "the flight (default 20)")
    ap.add_argument("--eval-nomad-altitudes", nargs="+", type=int, default=[10],
                    metavar="M",
                    help="altitudes used for the actor prototypes (default 10: "
                         "the only altitude where a garment colour is really in "
                         "the pixels)")
    ap.add_argument("--eval-nomad-profiles", default=None, metavar="JSON",
                    help="captions file (default <crop>/nomad_profiles.json; "
                         "pass vlm/nomad_vlm_profiles.json to score the "
                         "VLM-derived profiles instead)")
    # Weights & Biases logging (opt-in)
    ap.add_argument("--wandb", action="store_true",
                    help="log config, per-epoch loss and retrieval metrics to "
                         "Weights & Biases")
    ap.add_argument("--wandb-project", default="sar-person-clip-lora")
    ap.add_argument("--wandb-entity", default=None,
                    help="W&B team/user (default: your default entity)")
    ap.add_argument("--wandb-run-name", default=None,
                    help="run name (default: W&B auto-generates one)")
    ap.add_argument("--wandb-mode", default="online",
                    choices=["online", "offline", "disabled"],
                    help="'offline' logs locally (sync later); 'disabled' is a no-op")
    args = ap.parse_args()

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s %(name)s: %(message)s")

    # resolve --lora-alpha ('rank' -> alpha == rank -> scaling exactly 1.0)
    if str(args.lora_alpha).lower() in ("rank", "r", "auto"):
        args.lora_alpha = args.lora_rank
    else:
        args.lora_alpha = int(args.lora_alpha)
    if args.full_finetune:
        log.info("[mode] full fine-tune: ALL backbone params trainable "
                 "(LoRA rank/alpha ignored). Ensure --lr <= 5e-5 and "
                 "--epochs <= 30 to avoid catastrophic forgetting.")
        # Default grad-clip to 1.0 for full FT unless the user explicitly set it
        if args.grad_clip == 0.0:
            args.grad_clip = 1.0
            log.info("[mode] grad-clip auto-set to 1.0 for full fine-tune "
                     "(pass --grad-clip 0 to disable explicitly)")
    else:
        log.info("[lora] rank=%d alpha=%d -> scaling=%.3f", args.lora_rank,
                 args.lora_alpha, args.lora_alpha / max(args.lora_rank, 1))

    # Pin the GPU BEFORE torch is imported (encoders import it lazily). Setting
    # CUDA_VISIBLE_DEVICES keeps the visible device as cuda:0 for torch, so
    # SAR_Person's AMP check (`self.device == "cuda"`) stays true — unlike
    # --device cuda:1, which silently disables mixed precision. An env var the
    # user already exported (e.g. in ~/.bashrc) is respected and not overridden.
    if str(args.gpu).lower() != "all" and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    elif "CUDA_VISIBLE_DEVICES" in os.environ and \
            str(os.environ["CUDA_VISIBLE_DEVICES"]) != str(args.gpu):
        log.warning("[gpu] --gpu %s IGNORED: CUDA_VISIBLE_DEVICES=%s is already "
                    "set in the environment and wins. `unset "
                    "CUDA_VISIBLE_DEVICES` first if you meant --gpu.",
                    args.gpu, os.environ["CUDA_VISIBLE_DEVICES"])
    if args.encoder == "clip":
        log.info("[gpu] CUDA_VISIBLE_DEVICES=%s",
                 os.environ.get("CUDA_VISIBLE_DEVICES", "(unset: all GPUs)"))

    # 0 / negative == no limit. Without this, `--limit 0` sampled the pool down
    # to zero pairs and the run died deep inside the encoder with
    # "need at least one array to concatenate" -- an error that says nothing
    # about the flag that caused it.
    if args.limit is not None and args.limit <= 0:
        args.limit = None
        log.info("[data] --limit 0: using the FULL split of every dataset")

    out = Path(args.out)
    # Before 2026-09-18 this was a bare mkdir(exist_ok=True): reusing the --out of
    # an earlier run SILENTLY overwrote its lora_ckpt.pt and metrics.json -- no
    # error, no warning, discovered only after the GPU job had finished. For the
    # runs reported in the paper that would be permanent data loss.
    if (out / "lora_ckpt.pt").exists() and not args.overwrite:
        raise SystemExit(
            f"!! --out already holds a checkpoint: {out / 'lora_ckpt.pt'}\n"
            f"   Continuing would OVERWRITE it. Choose one:\n"
            f"     --out {out}_v2   keep the old run  (recommended)\n"
            f"     --overwrite      overwrite on purpose")
    # .frozen: marker written when a run's results are frozen for the paper
    if (out / ".frozen").exists():
        raise SystemExit(
            f"!! --out is FROZEN: {out / '.frozen'}\n"
            f"   {((out / '.frozen').read_text().splitlines() or [''])[0]}\n"
            f"   -> choose another --out, or delete the .frozen marker on purpose")
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "lora_ckpt.pt"
    rng = np.random.default_rng(args.seed)

    from .encoders import get_encoder
    enc = (get_encoder("clip", pretrain=args.pretrain, device=args.device,
                       img_size=tuple(args.img_size),
                       stride=int(args.pretrain.rsplit("/", 1)[-1])
                       if "/" in args.pretrain else 16)
           if args.encoder == "clip" else get_encoder("mock"))

    if args.init_ckpt:
        if args.encoder == "clip":
            # strict here too: a stage-2 curriculum run that silently starts
            # from zero-shot instead of the stage-1 adapter is the same bug as
            # at eval time, only more expensive -- you find out after the GPU
            # hours are spent. Use --lora-rank matching the checkpoint.
            ep0 = enc.load_trainable(args.init_ckpt, lora_rank=args.lora_rank,
                                     lora_alpha=args.lora_alpha, strict=True,
                                     full_finetune=args.full_finetune or None)
            log.info("[resume] adapter initialised from %s (saved epoch=%s)",
                     args.init_ckpt, ep0)
        else:
            log.warning("[resume] --init-ckpt ignored for encoder=%s",
                        args.encoder)

    # ---- what the init checkpoint was (added 2026-10-02) ------------------ #
    # base_kind = "full" means: this run trains a LoRA adapter ON TOP of a full
    # fine-tuned backbone (e.g. --init-ckpt Docc without --full-finetune). The
    # adapter saved below is then meaningless on OpenAI CLIP; eval_protocol and
    # SAR_Pipeline read this key and refuse it until it is merged with
    # scripts/export_merged_ckpt.py.
    base_kind = None
    if args.init_ckpt and args.encoder == "clip":
        from .encoders import is_full_finetune_ckpt
        init_full = is_full_finetune_ckpt(args.init_ckpt)
        if init_full and not args.full_finetune:
            base_kind = "full"
            log.warning("[resume] LoRA on top of a FULL backbone (%s): the "
                        "saved adapter needs that backbone. Deploy/evaluate "
                        "the merged file from scripts/export_merged_ckpt.py.",
                        args.init_ckpt)
        elif not init_full:
            base_kind = "lora"
        # The input size is part of the model. A full checkpoint at another
        # size fails to load (positional embedding); a LoRA one loads fine and
        # then trains on an input distribution it never saw -- warn loudly.
        _init_run = Path(args.init_ckpt).with_name("metrics.json")
        if _init_run.is_file():
            try:
                _isz = json.loads(_init_run.read_text(encoding="utf-8")).get(
                    "img_size")
            except (OSError, ValueError):
                _isz = None
            if _isz and list(_isz) != list(args.img_size):
                log.warning("[resume] --img-size %s differs from the init "
                            "checkpoint's run (%s). Pass --img-size %s %s "
                            "unless the change is deliberate.",
                            list(args.img_size), list(_isz), _isz[0], _isz[-1])

    mode = "joint" if (args.joint or len(args.datasets) == 1) else "sequential"
    log.info("[plan] datasets=%s mode=%s encoder=%s", args.datasets, mode,
             args.encoder)

    # ---- Weights & Biases (opt-in) --------------------------------------- #
    run = wandb = None
    gstep = {"n": 0}          # monotonic step shared across stages/trials/epochs

    def on_epoch_factory(stage, trial):
        """Return an on_epoch callback that streams the loss curve to W&B."""
        def cb(ep, avg_loss):
            gstep["n"] += 1
            run.log({"train/loss": avg_loss, "train/epoch": ep,
                     "train/trial": trial, "train/stage": stage},
                    step=gstep["n"])
        return cb if run is not None else None

    def on_eval_factory(stage, trial):
        """Stream the mid-training validation curve to W&B.

        This is the curve that shows whether --epochs was too many: the loss
        alone keeps falling long after retrieval has peaked, which is exactly
        why the last epoch was the wrong thing to save.
        """
        def cb(ep, metrics, is_best):
            d = {f"val/{stage}/{k}": v for k, v in metrics.items()
                 if isinstance(v, (int, float))}
            d["val/epoch"] = ep
            d["val/is_best"] = int(bool(is_best))
            d["val/trial"] = trial
            run.log(d, step=gstep["n"])
        return cb if run is not None else None

    def log_stage_metrics(stage, m_before, m_after):
        if run is None:
            return
        d = {}
        if m_before is not None:
            d.update({f"eval/{stage}/before/{k}": v for k, v in m_before.items()})
        d.update({f"eval/{stage}/after/{k}": v for k, v in m_after.items()})
        run.log(d, step=gstep["n"])
        for k, v in m_after.items():          # headline numbers in the run summary
            run.summary[f"{stage}/{k}"] = v

    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_run_name, mode=args.wandb_mode,
            config={
                "datasets": args.datasets, "mode": mode,
                "encoder": args.encoder, "pretrain": args.pretrain,
                "epochs": args.epochs, "trials": args.trials,
                "tuning": "full" if args.full_finetune else "lora",
                "lora_rank": args.lora_rank if not args.full_finetune else None,
                "lora_alpha": args.lora_alpha if not args.full_finetune else None,
                "lora_scaling": (args.lora_alpha / max(args.lora_rank, 1)
                                 if not args.full_finetune else None),
                "grad_clip": args.grad_clip,
                "id_loss_weight": args.id_loss_weight,
                "mlm_loss_weight": args.mlm_loss_weight,
                "cmt_depth": args.cmt_depth,
                "aug": args.aug,
                "sampler": args.sampler,
                "temp": args.temp,
                "eval_ids": args.eval_ids,
                "lr": args.lr, "batch": args.batch,
                "lr_schedule": args.lr_schedule,
                "warmup_epochs": args.warmup_epochs,
                "eval_every": args.eval_every,
                "select_metric": args.select_metric,
                "hard_neg_weight": args.hard_neg_weight, "margin": args.margin,
                # IRRA optimizer parity knobs -- without these in metrics.json
                # two runs with different aux-head LRs look identical on disk.
                "lr_factor": args.lr_factor,
                "bias_lr_factor": args.bias_lr_factor,
                "weight_decay": args.weight_decay,
                "weight_decay_bias": args.weight_decay_bias,
                "optimizer": args.optimizer,
                # was `"temp": 0.07` -- a hard-coded duplicate of the "temp"
                # key set above, so metrics.json reported 0.07 for every run
                # whatever --temp said. Fixed 2026-09-15.
                "amp": not args.no_amp,
                "limit": args.limit, "eval_size": args.eval_size,
                "seed": args.seed, "init_ckpt": args.init_ckpt,
            },
        )
        log.info("[wandb] logging to %s (%s)", run.url, args.wandb_mode)

    # NOMAD actor matrix, scored BEFORE training for the same reason as the
    # fixed benchmark below: without the baseline, a pretty heatmap says
    # nothing about what THIS run changed.
    nomad_split = _nomad_eval_split(args)
    S_nomad_before = None
    if nomad_split:
        S_nomad_before = _nomad_actor_matrix(enc, args, nomad_split)

    # Fixed benchmark, scored BEFORE any training so the after-minus-before
    # delta is attributable to this run rather than to zero-shot CLIP.
    bench_pairs, bench_before = None, None
    if args.eval_on:
        bench_pairs = load_benchmark_pairs(args.eval_on, args)
        bench_before, _ = _retrieval_metrics(enc, bench_pairs, args.enc_batch)
        log.info("[benchmark] %s BEFORE: R@1=%.4f R@5=%.4f", args.eval_on,
                 bench_before["i2t_R@1"], bench_before.get("i2t_R@5", float("nan")))

    t0 = time.time()
    stages = []
    select_info = []          # per (stage, trial): which epoch was selected

    if mode == "joint":
        pool = []
        for name in args.datasets:
            pool += load_dataset_pairs(name, args, args.limit)
        log.info("[data] pooled %d pair(s) from %s", len(pool), args.datasets)
        if not pool:
            raise SystemExit(
                f"no (image, caption) pairs from {args.datasets}. Check "
                f"--limit (0 = unlimited; a positive value smaller than the "
                f"split truncates it), the split filter, and that the images "
                f"resolve under --data-root {args.data_root}.")
        train_pairs, eval_pairs = _split_train_eval(pool, args.eval_size, rng,
                                                   eval_ids=args.eval_ids)
        m_before, S_before = _retrieval_metrics(enc, eval_pairs, args.enc_batch)
        stage_name = "+".join(args.datasets)
        best = None
        for tr in range(args.trials):
            sel_info = {}
            m_after, S_after = _train_once(
                enc, train_pairs, eval_pairs, args, ckpt, args.seed + tr,
                on_epoch=on_epoch_factory(stage_name, tr),
                on_eval=on_eval_factory(stage_name, tr), info=sel_info)
            select_info.append({"stage": stage_name, "trial": tr, **sel_info})
            if best is None or m_after["i2t_R@1"] > best[0]["i2t_R@1"]:
                best = (m_after, S_after)
        m_after, S_after = best
        log_stage_metrics(stage_name, m_before, m_after)
        stages.append({"stage": stage_name, "n_train":
                       len(train_pairs), "n_eval": len(eval_pairs),
                       "metrics_before": m_before, "metrics_after": m_after})
        _save_figs(out, S_before, S_after, eval_pairs)
    else:
        # staged: each dataset trains the SAME adapter, carried forward
        for si, name in enumerate(args.datasets):
            pairs = load_dataset_pairs(name, args, args.limit)
            train_pairs, eval_pairs = _split_train_eval(
                pairs, args.eval_size, rng, eval_ids=args.eval_ids)
            m_before, _ = _retrieval_metrics(enc, eval_pairs, args.enc_batch)
            best = None
            for tr in range(args.trials):
                sel_info = {}
                m_after, S_after = _train_once(
                    enc, train_pairs, eval_pairs, args, ckpt, args.seed + tr,
                    on_epoch=on_epoch_factory(name, tr),
                    on_eval=on_eval_factory(name, tr), info=sel_info)
                select_info.append({"stage": name, "trial": tr, **sel_info})
                if best is None or m_after["i2t_R@1"] > best[0]["i2t_R@1"]:
                    best = (m_after, S_after)
            m_after, S_after = best
            log_stage_metrics(name, m_before, m_after)
            snap = out / f"stage{si+1}_{name}_ckpt.pt"
            if ckpt.exists():
                shutil.copyfile(ckpt, snap)   # per-stage snapshot of the adapter
            log.info("[stage %d/%d %s] R@1 %.4f -> %.4f (train=%d eval=%d)",
                     si + 1, len(args.datasets), name, m_before["i2t_R@1"],
                     m_after["i2t_R@1"], len(train_pairs), len(eval_pairs))
            stages.append({"stage": name, "n_train": len(train_pairs),
                           "n_eval": len(eval_pairs), "snapshot": str(snap),
                           "metrics_before": m_before, "metrics_after": m_after})
        _save_figs(out, None, S_after, eval_pairs)   # figures for the last stage

    # ---- NOMAD actor matrix, after training ------------------------------ #
    nomad_eval = None
    res = _nomad_actor_matrix(enc, args, nomad_split, full=True) \
        if nomad_split else None
    if res:
        from . import eval_nomad_actors as na
        nomad_eval = na.save_artifacts(
            res["S"], res["actors"], res["prompts"], res["crops"], out,
            tag=(f"{'+'.join(args.datasets)} full-ft"
                 if args.full_finetune else
                 f"{'+'.join(args.datasets)} r{args.lora_rank}"),
            S_before=S_nomad_before, suffix="_nomad")
        nomad_eval["actor_split"] = nomad_split
        nomad_eval["profiles"] = args.eval_nomad_profiles or "metadata"
        # The protocol block. n_crops alone does not say WHY a prototype has
        # 20 crops rather than 93, and that difference is worth ~4 points of
        # R@1 -- enough to read as a model result when it is a sampling one.
        # SAR_Pipeline's eval_nomad.py writes the same four fields, so two
        # reports can be diffed instead of trusted.
        nomad_eval["altitudes"] = args.eval_nomad_altitudes
        nomad_eval["max_crops"] = args.eval_nomad_crops
        nomad_eval["min_visibility"] = args.nomad_min_vis
        nomad_eval["prompt_style"] = args.nomad_prompt_style
        m, m0 = nomad_eval["metrics"], nomad_eval["metrics_zeroshot"]
        log.info("[nomad] %d actor(s), split=%s | t2i R@1 %.4f -> %.4f  "
                 "MRR %.3f -> %.3f  gap %+.4f -> %+.4f",
                 nomad_eval["n_actors"], nomad_split, m0["t2i_R@1"],
                 m["t2i_R@1"], m0["t2i_MRR"], m["t2i_MRR"],
                 m0["pos_neg_gap"], m["pos_neg_gap"])
        if m["n_unique_captions"] < nomad_eval["n_actors"]:
            log.warning("[nomad] only %d/%d captions are unique; compare "
                        "t2i_R@1_caption=%.4f against the strict %.4f",
                        m["n_unique_captions"], nomad_eval["n_actors"],
                        m["t2i_R@1_caption"], m["t2i_R@1"])
        if run is not None:
            run.log({f"nomad_actors/before/{k}": v for k, v in m0.items()},
                    step=gstep["n"])
            run.log({f"nomad_actors/after/{k}": v for k, v in m.items()},
                    step=gstep["n"])
            for k, v in m.items():
                run.summary[f"nomad_actors/{k}"] = v
            for name in ("heatmap_after_nomad.png",
                         "heatmap_before_after_nomad.png"):
                if (out / name).is_file():
                    run.log({f"nomad_actors/{Path(name).stem}":
                             wandb.Image(str(out / name))}, step=gstep["n"])

    # ---- fixed benchmark, after training -------------------------------- #
    benchmark = None
    if args.eval_on:
        bench_after, _ = _retrieval_metrics(enc, bench_pairs, args.enc_batch)
        benchmark = {"name": args.eval_on, "n_pairs": len(bench_pairs),
                     "before": bench_before, "after": bench_after,
                     "delta_R@1": round(bench_after["i2t_R@1"]
                                        - bench_before["i2t_R@1"], 4)}
        log.info("[benchmark] %s AFTER : R@1=%.4f (delta %+.4f) on %d image(s)",
                 args.eval_on, bench_after["i2t_R@1"], benchmark["delta_R@1"],
                 len(bench_pairs))
        if run is not None:
            run.log({f"benchmark/{args.eval_on}/before/{k}": v
                     for k, v in bench_before.items()}, step=gstep["n"])
            run.log({f"benchmark/{args.eval_on}/after/{k}": v
                     for k, v in bench_after.items()}, step=gstep["n"])
            for k, v in bench_after.items():
                run.summary[f"benchmark_{args.eval_on}/{k}"] = v
            run.summary[f"benchmark_{args.eval_on}/delta_R@1"] = \
                benchmark["delta_R@1"]

    report = {"datasets": args.datasets, "mode": mode, "encoder": args.encoder,
              "pretrain": args.pretrain,
              "img_size": list(args.img_size), "loss": args.loss,
              "epochs": args.epochs,
              "tuning": "full" if args.full_finetune else "lora",
              "lora_rank": args.lora_rank if not args.full_finetune else None,
              "lora_alpha": args.lora_alpha if not args.full_finetune else None,
              "lora_scaling": (round(args.lora_alpha / max(args.lora_rank, 1), 4)
                               if not args.full_finetune else None),
              "grad_clip": args.grad_clip,
              "lr": args.lr, "batch": args.batch, "limit": args.limit,
              "lr_schedule": args.lr_schedule,
              "warmup_epochs": args.warmup_epochs,
              "eval_every": args.eval_every,
              "select_metric": args.select_metric,
              "seed": args.seed, "init_ckpt": args.init_ckpt,
              "base_kind": base_kind,
              # Everything that defines the run but was never written here
              # before 2026-10-02 -- runs that differed only in occ_prob were
              # indistinguishable on disk because it was not recorded.
              "train_config": {
                  "base_kind": base_kind,
                  "loss": args.loss, "sdm_temp": args.sdm_temp,
                  "temp": args.temp, "num_instance": args.num_instance,
                  "sampler": args.sampler, "aug": args.aug,
                  "occ_prob": args.occ_prob,
                  "occ_variants": list(args.occ_variants),
                  "occ_levels": list(args.occ_levels),
                  "id_loss_weight": args.id_loss_weight,
                  "mlm_loss_weight": args.mlm_loss_weight,
                  "cmt_depth": args.cmt_depth,
                  "vis_loss_weight": getattr(args, "vis_loss_weight", None),
                  "ortho_weight": getattr(args, "ortho_weight", None),
                  "lr_factor": args.lr_factor,
                  "bias_lr_factor": args.bias_lr_factor,
                  "weight_decay": args.weight_decay,
                  "weight_decay_bias": args.weight_decay_bias,
                  "optimizer": args.optimizer,
                  "hard_neg_weight": args.hard_neg_weight,
                  "eval_ids": args.eval_ids, "eval_size": args.eval_size,
                  "nomad_prompt_style": args.nomad_prompt_style,
                  "nomad_altitudes": args.nomad_altitudes,
                  "nomad_min_vis": args.nomad_min_vis,
                  "nomad_activities": args.nomad_activities,
                  "lora_attn": args.lora_attn,
              },
              "seconds": round(time.time() - t0, 1),
              "checkpoint": str(ckpt) if args.encoder == "clip" else None,
              "checkpoint_selection": select_info,
              "benchmark": benchmark,
              "nomad_actor_eval": nomad_eval,
              "stages": stages}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    log.info("[done] mode=%s -> %s", mode, out)
    if args.encoder == "clip":
        if args.full_finetune:
            log.info("[done] checkpoint: %s  (full fine-tune, reload with "
                     "build_CLIP_from_openai_pretrained + torch.load)", ckpt)
        else:
            log.info("[done] checkpoint: %s  (SAR_Pipeline: --lora-ckpt %s "
                     "--lora-rank %d --lora-alpha %s)",
                     ckpt, ckpt, args.lora_rank, args.lora_alpha)
        for si_ in select_info:
            if si_.get("selected"):
                log.info("[done] %s trial %s: best epoch %s/%s (%s=%.4f)%s",
                         si_.get("stage"), si_.get("trial"),
                         si_.get("best_epoch"), args.epochs,
                         args.select_metric, si_.get("best_score", float("nan")),
                         "  <-- ran too long, cut --epochs"
                         if si_.get("best_epoch", 0) < args.epochs - 1 else "")

    if run is not None:
        run.summary["seconds"] = report["seconds"]
        run.summary["final_R@1"] = stages[-1]["metrics_after"]["i2t_R@1"]
        if select_info and select_info[-1].get("selected"):
            run.summary["best_epoch"] = select_info[-1].get("best_epoch")
            run.summary["best_score"] = select_info[-1].get("best_score")
        # keep the run's output files (metrics.json + heatmaps) alongside the run
        art = wandb.Artifact(f"{run.id}-outputs", type="training-output")
        art.add_dir(str(out))
        run.log_artifact(art)
        run.finish()
    print(json.dumps({"datasets": args.datasets, "mode": mode,
                      "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                      "checkpoint": report["checkpoint"],
                      "final_R@1": stages[-1]["metrics_after"]["i2t_R@1"],
                      "benchmark_R@1": (benchmark["after"]["i2t_R@1"]
                                        if benchmark else None),
                      "nomad_t2i_R@1": (nomad_eval["metrics"]["t2i_R@1"]
                                        if nomad_eval else None)},
                     indent=2))


# --------------------------------------------------------------------------- #
# NOMAD actor-level check (see sar_person.eval_nomad_actors)                   #
# --------------------------------------------------------------------------- #

def _nomad_eval_split(args) -> "str | None":
    """Which NOMAD actors this run may honestly be scored on -- or None.

    The whole point of ``auto``: a run that trained on the ``nomad`` preset has
    already seen 80 of the 100 actors WITH their captions, so scoring all 100
    measures memorisation. Only the 20 held-out actors are a retrieval result.
    A run that never touched NOMAD can use all 100, which is both a harder
    gallery and a better picture.
    """
    if args.eval_nomad == "off":
        return None
    crop_root = Path(args.data_root) / PRESETS["nomad"]["crop_root"]
    if not (crop_root / "annotations.json").is_file():
        if args.eval_nomad != "auto":
            log.warning("[nomad] --eval-nomad %s but %s is missing; skipping",
                        args.eval_nomad, crop_root / "annotations.json")
        return None
    if args.eval_nomad != "auto":
        return args.eval_nomad
    if "nomad" in args.datasets:
        log.info("[nomad] 'nomad' is in --datasets -> scoring the 20 held-out "
                 "val actors only (--eval-nomad all would report memorisation)")
        return "val"
    return "all"


def _nomad_actor_matrix(enc, args, split: str, full: bool = False):
    """Actor prototypes x profile prompts for the current encoder state.

    Returns the matrix alone by default (the before-training baseline) or the
    whole ``build_actor_matrix`` payload when ``full``. Failures are logged and
    swallowed: this is a diagnostic bolted onto a training run, and a missing
    profiles file must never destroy hours of finished training.
    """
    from . import eval_nomad_actors as na
    crop_root = Path(args.data_root) / PRESETS["nomad"]["crop_root"]
    try:
        res = na.build_actor_matrix(
            enc, crop_root, profiles=args.eval_nomad_profiles,
            prompt_style=args.nomad_prompt_style,
            altitudes=args.eval_nomad_altitudes,
            max_per_actor=args.eval_nomad_crops, split=split,
            batch=args.enc_batch)
    except (SystemExit, OSError, KeyError, ValueError) as exc:
        log.warning("[nomad] actor eval skipped: %s", exc)
        return None
    return res if full else res["S"]


def _save_figs(out: Path, S_before, S_after, eval_pairs):
    k = min(20, len(eval_pairs))
    gt = list(range(k))
    if S_before is not None:
        mv.save_single(S_before[:k, :k], gt, out / "heatmap_before.png",
                       "Before (zero-shot)")
    mv.save_single(S_after[:k, :k], gt, out / "heatmap_after.png", "After (LoRA)")
    mv.save_csv(S_after, [p["image_id"] for p in eval_pairs],
                str(out / "similarity_after.csv"))


if __name__ == "__main__":
    main()
