# Manifests

Dataset-free descriptions of every image set used in the paper. They contain
dataset-relative image IDs, identity labels and hashes. They contain **no pixels and
no caption text**. All files are gzip-compressed JSON.

| folder | file | content | checked by |
|---|---|---|---|
| `evalsets/` | `cuhk`, `icfg`, `rstp`, `nomad` `.json.gz` | the test gallery `[image_id, pid]` and queries `[image_id, pid, sha1(caption)]`, in evaluation order | `scripts/verify_evalsets.py` |
| `train/` | `cuhk_author_train.json.gz` | the 34,054 CUHK-PEDES training images `[image_id, person_id, [sha1(caption), ...]]`, in training-file order, plus the SHA-256 of the annotation files used | `scripts/verify_training_data.py` |
| `train/` | `nomad_crop.json.gz` | all 9,277 NOMAD person crops (id, actor, split, altitude, visibility, source frame, source box, crop size), the build flags, the 20 validation actors and a SHA-1 of each actor's description | `scripts/verify_training_data.py` |
| `occluded/` | `<bench>_<geometry>_lNN.json.gz` (45 files) | for a fixed sample of 300 gallery images per ORP cell: gallery index, image ID, occluder seed, SHA-256 of the occluded raw RGB array and of the JPEG, fraction of pixels changed, image size | `scripts/verify_occlusion.py` |
| `occluded_oof/` | `<tag>.json.gz` (6 files) | out-of-family galleries: which gallery images are occluded, selection mode and occluder scale | regenerate with `configs/eval/oof_mgcc.yaml` |

Gallery and query sizes: CUHK-PEDES 3,074 / 6,156; ICFG-PEDES 19,848 / 19,848;
RSTPReid 1,000 / 2,000; NOMAD-val 1,956 / 20.

Caption hashes are `sha1(caption.strip().encode("utf-8"))`. Image IDs are relative to
the image folders listed in `docs/DATASETS.md`.

The OOF manifests do not yet carry pixel hashes: the rendered OOF galleries are on
the training server. Running `python tools/export_manifests.py --oof --result-root
<Result> --out manifests` there adds them.
