# Datasets: how to obtain, lay out and verify them

**No images and no caption text are distributed with this repository.** All datasets
used in the paper are licensed by their authors for research use and may not be
redistributed. Please obtain each one from its official source under its own terms.
What this repository ships instead:

* **evaluation manifests** (`manifests/evalsets/`): the exact gallery and query lists,
  in evaluation order, as dataset-relative image IDs, identity labels and a SHA-1 of
  every query caption;
* **training manifests** (`manifests/train/`): the exact training images (CUHK-PEDES)
  and crops (NOMAD), with identity labels and caption SHA-1s;
* **occlusion manifests** (`manifests/occluded/`, `manifests/occluded_oof/`): per-image
  occluder seeds and pixel hashes (see [OCCLUSION.md](OCCLUSION.md));
* **verification scripts** that prove your local copy yields exactly the data used in
  the paper.

## Overview

| dataset | role in the paper | split used | size used | official source |
|---|---|---|---|---|
| CUHK-PEDES | training (D, D+A, …) and evaluation | official train / official test | train 34,054 images, 11,003 ids, 68,126 captions; test 3,074 images, 1,000 ids, 6,156 queries | <https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description> |
| ICFG-PEDES | evaluation only (never trained on) | official test | 19,848 images, 1,000 ids, 19,848 queries | <https://github.com/zifyloo/SSAN> |
| RSTPReid | evaluation only (never trained on) | official test | 1,000 images, 200 ids, 2,000 queries | <https://github.com/NjtechCVLab/RSTPReid-Dataset> |
| NOMAD | aerial Stage 2 training (80 actors) and NOMAD-val (20 actors) | actor-level 80/20 split (ours) | 9,277 person crops: 7,321 train, 1,956 val | <https://github.com/ArtRuss/NOMAD> (Bernal et al., WACV 2024) |
| MGCC occlusion library (OIL) | out-of-family occluders (evaluation only) | all 60 cut-outs | 15 classes × 4 PNGs, ~700 kB | `python scripts/fetch_oil.py` (from MGCC's public repository) |

The CLIP ViT-B/16 initial weights are downloaded automatically from OpenAI's public URL
on first use (`model/clip_model.py`). On a machine behind a TLS-intercepting proxy,
`bash scripts/fetch_clip_weights.sh ViT-B/16` installs them into `~/.cache/clip` after
checking their SHA-256.

## Directory layout

All scripts take `--data-root`, the folder that contains the datasets (by default
`../Dataset`, a sibling of this repository). The layout the code expects is:

```
Dataset/
├── CUHK-PEDES/
│   ├── reid_raw.json                 # official annotation with the train/val/test split
│   ├── imgs/                         # official images: CUHK01/ CUHK03/ Market/ test_query/ train_query/
│   └── annotations/train.jsonl       # written by scripts/prepare_cuhk.py --rebuild-train
├── ICFG-PEDES/
│   └── ICFG-PEDES/
│       ├── ICFG-PEDES.json           # official annotation (named ICFG_PEDES.json in some copies: rename)
│       └── imgs/{train,test}/...
├── RSTPReid/
│   ├── imgs/                         # official images
│   └── data_captions.json            # official annotation
│      (or, alternatively, data/*.parquet: the parquet export the paper's loader reads)
├── NOMAD/
│   ├── annotations/{metadata.json, annotations.json, ...}   # official
│   ├── images/ActorNNN/ActorNNN_aXX/...                     # official frames (5472 x 3078)
│   └── crop/                         # written by scripts/prepare_nomad.py
│       ├── images/ActorNNN/*.jpg
│       ├── annotations.json
│       ├── nomad_profiles.json
│       └── build_report.json
└── OIL/                              # written by scripts/fetch_oil.py
```

If your copies live elsewhere, symbolic links are fine. The image IDs in the manifests
are relative to `CUHK-PEDES/imgs/`, `ICFG-PEDES/ICFG-PEDES/imgs/`, the RSTPReid image
folder and `NOMAD/crop/images/`, respectively.

## Per dataset

### CUHK-PEDES

1. Request the images from the authors (see their repository; academic use only) and
   download `reid_raw.json`, which carries the official train/val/test split.
2. Place them as above, then write the training file:
   ```bash
   python scripts/prepare_cuhk.py --data-root ../Dataset --rebuild-train
   ```
   This writes `CUHK-PEDES/annotations/train.jsonl` (34,054 images, 68,126 captions)
   from the official captions. It is the input of the `cuhk_author` training preset.
3. Verify:
   ```bash
   python scripts/verify_evalsets.py      --data-root ../Dataset --benchmarks cuhk
   python scripts/verify_training_data.py --data-root ../Dataset --only cuhk
   ```

The paper uses the **official** test split (`split_source: official` in
`manifests/evalsets/cuhk.json.gz`). `scripts/prepare_cuhk.py` can also reconstruct a
split when only the 2017 `caption_all.json` is available. Such a split is labelled
`reconstructed`, and numbers on it are **not** comparable to the paper.

Checkpoints are selected on 300 identities held out from the **training** split
(`--eval-ids 300`); the CUHK-PEDES test split is never used for model selection.

### ICFG-PEDES

Obtain it from the SSAN authors. Only the test split is used (all 19,848 images and
captions; no truncation). Verify with
`python scripts/verify_evalsets.py --data-root ../Dataset --benchmarks icfg`.

### RSTPReid

Obtain it from the authors. Two layouts are supported:

* **Official release** (`imgs/` + `data_captions.json`): build the evaluation set in
  the paper's order. Each caption is checked against the released SHA-1:
  ```bash
  python scripts/prepare_rstp_official.py --data-root ../Dataset --cache runs/evalsets
  ```
  Then use `--out-root runs` with `tools/run_config.py`, or
  `--evalset-cache runs/evalsets` with the modules directly.
* **Parquet export** (`RSTPReid/data/*.parquet` with columns `image`, `captions`, `id`,
  `file_path`, `split`; this is what the paper's loader `sar_person.eval_sets._load_rstp`
  reads; needs `pyarrow`): nothing to prepare. Verify with
  `python scripts/verify_evalsets.py --data-root ../Dataset --benchmarks rstp`.

RSTPReid is used for evaluation only.

### NOMAD

1. Download NOMAD from its repository (frames, `annotations/`).
2. Crop the people and build the actor-level split:
   ```bash
   python scripts/prepare_nomad.py --workers 8
   ```
   The defaults are those of the paper: altitudes 10 m and 30 m, visibility ≥ 50%,
   box padding 12%, minimum crop side 24 px, actor-level split 80/20 with seed 0. The
   20 validation actors are listed in `manifests/train/nomad_crop.json.gz`
   (`build_flags.split.val_actor_ids`). Each actor has one appearance description
   (`prompt_default`), generated from NOMAD's `metadata.json` by
   `sar_person/nomad_meta.py`.
3. Verify:
   ```bash
   python scripts/verify_training_data.py --data-root ../Dataset --only nomad
   python scripts/verify_evalsets.py      --data-root ../Dataset --benchmarks nomad
   ```

NOMAD-val (the 20 held-out actors, 1,956 crops) is never trained on. The aerial models
use their last epoch, so NOMAD-val is not used for checkpoint selection either.

### MGCC occlusion library (out-of-family occluders)

```bash
python scripts/fetch_oil.py --out ../Dataset/OIL
```

This downloads the 60 object cut-outs (15 classes × 4) that MGCC uses to occlude
galleries. `--from-local` copies them from a local clone instead.

## What the verification scripts guarantee

| script | checks | against |
|---|---|---|
| `scripts/verify_evalsets.py` | gallery images and order, identity labels, query order, query caption text (SHA-1) | `manifests/evalsets/<bench>.json.gz` |
| `scripts/verify_training_data.py` | training images/crops and order, identities, caption text (SHA-1), NOMAD split and boxes | `manifests/train/*.json.gz` |
| `scripts/verify_occlusion.py` | occluded pixels of a fixed sample of 300 gallery images per cell (SHA-256 of the raw RGB array) | `manifests/occluded/*.json.gz` |

If all three pass, your evaluation data, training data and occluded galleries are
identical to the paper's, and the released checkpoints should reproduce the per-query
results in `results/per_query/` up to GPU floating-point non-determinism.
