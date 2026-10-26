# Occlusion Retention Protocol (ORP): code, configurations and data manifests

Reproducibility package for the paper

> **Evaluating Occlusion Robustness in Text-to-Image Person Retrieval Beyond Rank-1 Accuracy**
> (KSC 2026 submission; authors withheld for double-blind review)

ORP measures how text-to-image person retrieval degrades under occlusion. It scores
every model on the same occluded galleries (3 occluder geometries × 5 coverages), and
reports how much of the clean **mINP** (the rank of the hardest correct image) is
retained, next to the usual Rank-1. The repository contains everything needed to
regenerate those galleries, retrain and re-score every model in the paper, and
recompute every number in it. **No dataset images or caption text are included**
(see [Datasets](#3-datasets)).

## Where each resource is

| resource | location |
|---|---|
| Dataset manifests (evaluation sets, training sets, occluded galleries) | [`manifests/`](manifests/README.md) |
| Occlusion generation code | [`sar_person/occlude_bench.py`](sar_person/occlude_bench.py) (`degrade`, `_seed_for`), [`sar_person/occlude_mals.py`](sar_person/occlude_mals.py) (foliage), [`sar_person/ogor.py`](sar_person/ogor.py) (real-object occluders); documented in [`docs/OCCLUSION.md`](docs/OCCLUSION.md) |
| Model checkpoints | 10 files, about 600 MB each; list, hashes and download in [`checkpoints/README.md`](checkpoints/README.md) |
| Training configurations | [`configs/train/`](configs/train/), one YAML per model with every argument, the selected epoch, the checkpoint SHA-256 and the original run |
| Evaluation configurations | [`configs/eval/`](configs/eval/): ORP grid, out-of-family occluders, NOMAD-val |
| Evaluation code | [`sar_person/eval_protocol.py`](sar_person/eval_protocol.py) (ranking and per-query metrics), [`analysis/orp.py`](analysis/orp.py) (retention, AURC, Div, bootstrap) |
| Per-query results behind every number | [`results/`](results/README.md) |
| Seed consistency and confidence intervals | [`docs/STATISTICS.md`](docs/STATISTICS.md), [`analysis/`](analysis/) |

## 1. Recompute the paper's numbers (no data, no GPU, about 10 s)

```bash
pip install numpy
python analysis/reproduce_paper.py     # 109 numbers printed in the paper -> 109/109 OK
python analysis/seed_consistency.py    # main findings checked for each training seed
python analysis/ci_stability.py        # how the 95% CIs behave (Monte-Carlo, identity vs. query resampling)
```

These scripts read only `results/` (per-query INP, AP and Rank-K hits for every model,
benchmark and occlusion cell) and `analysis/paper_values.json`.

## 2. Setup

Tested with Python 3.10 on Linux, one NVIDIA RTX 3090 (24 GB). A training run needs
about 13 GB of GPU memory with AMP.

```bash
bash scripts/setup_env.sh                       # venv + PyTorch (CUDA 12.4 wheels) + requirements
# or by hand:
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match your driver
pip install -r requirements.txt
```

The CLIP ViT-B/16 weights are downloaded from OpenAI on first use. Behind a
TLS-intercepting proxy, run `bash scripts/fetch_clip_weights.sh ViT-B/16`.

Scoring the *published IRRA checkpoint* (a baseline in Fig. 1 and Table 1) also needs
a clone of [IRRA](https://github.com/anosorae/IRRA) and its CUHK-PEDES `best.pth`.

## 3. Datasets

The datasets are licensed by their authors and are **not** redistributed. Obtain them
from the official sources, place them under one folder (default `../Dataset`) and run
the preparation scripts. Full instructions, the expected layout and the official links
are in **[`docs/DATASETS.md`](docs/DATASETS.md)**.

| dataset | used for | obtain from | prepare |
|---|---|---|---|
| CUHK-PEDES | training and evaluation | [authors](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) (`imgs/` + `reid_raw.json`) | `python scripts/prepare_cuhk.py --data-root ../Dataset --rebuild-train` |
| ICFG-PEDES | evaluation only | [authors](https://github.com/zifyloo/SSAN) | none |
| RSTPReid | evaluation only | [authors](https://github.com/NjtechCVLab/RSTPReid-Dataset) | `python scripts/prepare_rstp_official.py --data-root ../Dataset --cache runs/evalsets` |
| NOMAD | aerial training (80 actors) and NOMAD-val (20 actors) | [authors](https://github.com/ArtRuss/NOMAD) | `python scripts/prepare_nomad.py --workers 8` |
| MGCC occluder library | out-of-family occluders | public | `python scripts/fetch_oil.py --out ../Dataset/OIL` |

```
Dataset/
├── CUHK-PEDES/{reid_raw.json, imgs/, annotations/train.jsonl}
├── ICFG-PEDES/ICFG-PEDES/{ICFG-PEDES.json, imgs/}
├── RSTPReid/{data_captions.json, imgs/}          (or data/*.parquet)
├── NOMAD/{annotations/, images/, crop/}
└── OIL/
```

Then confirm that your copies are the paper's data. The manifests store image IDs,
identity labels and caption hashes, so these checks are exact:

```bash
python scripts/verify_evalsets.py      --data-root ../Dataset --benchmarks cuhk icfg rstp nomad
python scripts/verify_training_data.py --data-root ../Dataset
```

## 4. Occluded galleries

```bash
python tools/run_config.py occlude configs/eval/orp_ground.yaml --data-root ../Dataset --out-root runs
python tools/run_config.py occlude configs/eval/oof_mgcc.yaml   --data-root ../Dataset --out-root runs
python scripts/verify_occlusion.py --data-root ../Dataset --benchmarks cuhk rstp icfg
```

Each gallery image gets a seed derived from its dataset-relative path (SHA-1), so the
galleries are identical on every machine. `verify_occlusion.py` compares the raw-RGB
SHA-256 of 300 images per cell with `manifests/occluded/` and prints `ALL CELLS MATCH`
(45 cells). Geometry definitions, seeding, measured area per cell and the training-time
occlusion are in [`docs/OCCLUSION.md`](docs/OCCLUSION.md).

## 5. Evaluate the released checkpoints

```bash
python scripts/fetch_checkpoints.py                         # -> checkpoints/*.pt, SHA-256 verified

python tools/run_config.py eval configs/eval/orp_ground.yaml --model DA_s0 --data-root ../Dataset --out-root runs --gpu 0
python tools/run_config.py eval configs/eval/oof_mgcc.yaml   --model DA_s0 --data-root ../Dataset --out-root runs --gpu 0
python tools/run_config.py eval configs/eval/nomad_val.yaml  --model DA_nomad_s0 --data-root ../Dataset --out-root runs --gpu 0

python tools/compare_runs.py --out-root runs                # your numbers vs. the paper's, cell by cell
```

`--dry-run` prints the exact `python -m sar_person.eval_protocol ...` commands instead of
running them. `bash scripts/run_all.sh eval` scores every model of the paper.

`--model` takes a training-config name (`D`, `DA_s0`, `DA_s1`, `DA_s2`, `D_plus10`,
`DA_prime`, `IRRA_plus_A`, `DA_nomad_s0..2`), `clip_zs` (zero-shot CLIP) or
`irra_published` (needs the IRRA paths in `configs/eval/orp_ground.yaml`). Outputs go
to `runs/<model>/eval/<benchmark>_<cell>/metrics.json` and include per-query results.

## 6. Retrain

| config | paper name | init | what changes | GPU time (RTX 3090) |
|---|---|---|---|---|
| `D` | D | CLIP ViT-B/16 | 60 epochs on CUHK-PEDES, SDM + ID + MLM losses, IRRA optimiser settings, random erasing | 9.4 h |
| `DA_s0`, `DA_s1`, `DA_s2` | D+A, seeds 0/1/2 | D | +10 epochs, occlusion p = 0.5, c ∈ {0.25, 0.5, 0.7}, torso/cutout/foliage | 1.6 h each |
| `D_plus10` | D+10 | D | as D+A but **no** occlusion (extra-epochs control) | 1.6 h |
| `DA_prime` | D+A′ | D | as D+A, c ∈ {0.25, 0.55, 0.7} (disjoint from the test grid) | 1.7 h |
| `irra_init` → `IRRA_plus_A` | IRRA+A | published IRRA backbone | the D+A recipe | conversion + 1.6 h |
| `DA_nomad_s0..2` | D+A→NOMAD, seeds 0/1/2 | D+A (seed 0) | 10 epochs on 80 NOMAD actors, foliage only, lr 5e-6, **last** epoch | 0.2 h each |

```bash
python tools/run_config.py train configs/train/D.yaml     --data-root ../Dataset --out-root runs --gpu 0
python tools/run_config.py train configs/train/DA_s0.yaml --data-root ../Dataset --out-root runs --gpu 0
# ... or everything in dependency order:
bash scripts/run_all.sh train
```

`init_from` is resolved to your own retrained parent if it exists in `runs/`, and to the
released checkpoint otherwise. You can therefore retrain D+A from the released D
without retraining D. Each config's `checkpoint` block records which epoch the paper
used. Ground models are selected by text-to-image Rank-1 on 300 identities held out
from the CUHK-PEDES **training** split, never on a test set. Aerial models use their
last epoch.

**Determinism.** Data order, held-out identities, occlusion draws and augmentation are
seeded. GPU kernels and fp16 autocast are not, so a retrained model matches the
released one statistically, not bit for bit. The seed study in
[`docs/STATISTICS.md`](docs/STATISTICS.md) shows the size of run-to-run variation
(mINP-AURC SD 0.56 points over three seeds).

## 7. Statistics: confidence intervals and seeds

* **Confidence intervals** are 95% **paired, identity-level bootstrap** percentile
  intervals with B = 2,000. Test identities are resampled with replacement (1,000 for
  CUHK-PEDES and ICFG-PEDES, 200 for RSTPReid). The same resample is applied to both
  models in a comparison, and AURC is recomputed inside every replicate. NOMAD-val uses
  a bootstrap over its 20 actors (B = 10,000). The intervals reflect test-set sampling
  for a fixed model; training randomness is assessed separately through seeds.
* **Seeds.** D+A was trained with seeds 0, 1 and 2, and D+A→NOMAD with seeds 0, 1 and 2.
  Twelve of the thirteen checks in `analysis/seed_consistency.py` hold for every seed.
  For example, the mINP-AURC gain of D+A over D is +10.05 / +10.09 / +11.04 on
  CUHK-PEDES and +10.42 / +9.15 / +9.08 on RSTPReid, every CI excluding 0. The gain
  over the no-occlusion control D+10 is similar, at no clean Rank-1 cost. The one
  partial exception is reported: for seed 1 on RSTPReid, absolute mINP is below D's in
  2 of 15 cells, by 0.06 and 0.16 points.

Details, tables and the pending ICFG-PEDES seed runs: [`docs/STATISTICS.md`](docs/STATISTICS.md).

## 8. Repository layout

```
analysis/        ORP metrics + bootstrap (orp.py); reproduce_paper.py, seed_consistency.py, ci_stability.py
checkpoints/     README + SHA-256 list (weights downloaded by scripts/fetch_checkpoints.py)
configs/train/   one YAML per trained model (all arguments, selected epoch, hash, provenance)
configs/eval/    ORP grid (orp_ground), out-of-family occluders (oof_mgcc), NOMAD-val (nomad_val)
data/            CLIP BPE vocabulary
docs/            DATASETS.md, OCCLUSION.md, STATISTICS.md
manifests/       evaluation, training and occlusion manifests (no pixels, no captions)
model/           CLIP ViT-B/16 (adapted from OpenAI CLIP via IRRA), PETL helpers
results/         per-query results of every evaluated (model, benchmark, cell) + reproduced outputs
sar_person/      training (train_dataset.py, encoders.py), evaluation (eval_protocol.py, eval_sets.py),
                 occlusion (occlude_bench.py, occlude_mals.py, ogor.py), NOMAD tools
scripts/         dataset preparation, verification, checkpoint download, setup, run_all.sh
tests/           unit tests (pytest)
tools/           run_config.py (configs -> commands), compare_runs.py, maintainer exporters
```

## 9. Known limitations of this release

* **ICFG-PEDES for D+A seeds 1 and 2** has not been scored yet (seed 0 is in the
  paper). The commands are in `docs/STATISTICS.md`; the analysis scripts include these
  results automatically once they exist.
* **D+A (seed 0)** was launched by a script that was not archived. Its configuration
  was reconstructed from the run's own log header, which every argument in
  `configs/train/DA_s0.yaml` reproduces. The three single-change controls (D+10, D+A′,
  IRRA+A) and seeds 1–2 were launched by `run_revision.sh`, which compares each new log
  header with that D+A header and stops the run on any unintended difference. The
  recorded checks for D+10, D+A′, IRRA+A and seed 1 found no difference.
* **Checkpoint selection.** D and D+A (seed 0) validate on all 1,039 images of the 300
  held-out identities. The later runs cap validation at 1,000 images, the trainer's
  default. Both are recorded in the configs.
* **Published IRRA on RSTPReid and ICFG-PEDES** (Fig. 1) is kept as aggregate metrics
  only (`results/aggregate_only_cells.csv`), so no bootstrap CIs are computed for those
  cells.
* **Out-of-family galleries**: `manifests/occluded_oof/` lists which images are
  occluded, but not yet their pixel hashes.
* The original training logs record no PyTorch/CUDA versions.

## 10. License and citation

Code, configurations, manifests and results: MIT (see [`LICENSE`](LICENSE)), including
third-party code adapted from OpenAI CLIP and IRRA (both MIT). The datasets are not
covered and remain under their owners' terms.

```bibtex
@inproceedings{orp2026,
  title     = {Evaluating Occlusion Robustness in Text-to-Image Person Retrieval Beyond Rank-1 Accuracy},
  author    = {Anonymous},
  booktitle = {Korea Software Congress (KSC)},
  year      = {2026}
}
```
