# Occlusion generation

All occluders are produced by one function, `degrade()` in
`sar_person/occlude_bench.py`, from the clean image, the geometry, the level and a
per-image seed. Nothing else enters, so the occluded galleries can be regenerated
pixel-for-pixel from the released code, without distributing a single image.

## 1. The ORP test grid (evaluation)

3 geometries × 5 nominal coverages c ∈ {0.1, 0.2, 0.3, 0.4, 0.5} = 15 cells per
benchmark. In every cell, **every** gallery image is occluded. Queries, identity
labels and gallery size are unchanged, so any clean-to-occluded difference comes from
the occlusion alone.

| geometry | what is drawn | role of c |
|---|---|---|
| `torso` | one opaque black horizontal band across the full width, centred at half the image height | band height = c × image height |
| `cutout` | up to 3 opaque black rectangles; widths 0.25–0.6 × image width, placed uniformly at random | rectangles are added until their summed area reaches c × image area (each at most 0.7 × image height; overlaps and the 3-box cap make the destroyed area smaller than c at high c) |
| `foliage` | procedural vegetation: 3 + ⌊10c⌋ brown branches (lines 3–8 px wide) and 20 + ⌊90c⌋ semi-transparent green leaves (ellipses), clustered around a random horizontal band | c sets the number of leaves and branches, not an area; leaves are alpha-blended (alpha 180–245) |

**Seeding.** Each gallery image gets its own seed:
`seed = int.from_bytes(sha1(image_id).digest()[:4], "little") & 0x7FFFFFFF`
(`_seed_for` in `occlude_bench.py`). Here `image_id` is the dataset-relative path,
for example `test_query/p2647_s3376.jpg`. It is never an absolute path, so every
machine draws the same occluder. Python's `random.Random(seed)` drives `cutout` and
`foliage`, and `torso` is deterministic. An image has the same seed in every cell;
because the number of shapes depends on c, the layout itself still changes between
levels for `cutout` and `foliage`.

**Encoding.** Occluded images are written as JPEG, quality 95, and named
`<gallery index, 7 digits>_<image stem>.jpg`. Images outside the selection are read
from the original files, so they are never re-encoded.

**Generate** (once per benchmark; about 15 × gallery size images):

```bash
python tools/run_config.py occlude configs/eval/orp_ground.yaml --data-root ../Dataset --out-root runs
# equivalent to, per benchmark:
python -m sar_person.occlude_bench --benchmark cuhk --data-root ../Dataset \
    --variants torso cutout foliage --levels 0.1 0.2 0.3 0.4 0.5 \
    --out-root runs/occluded --quality 95 --workers 8
```

This writes `runs/occluded/images/<bench>_<geom>_lNN/` and one evaluation manifest per
cell, `runs/occluded/manifests/<bench>_<geom>_lNN.json`, which `eval_protocol --manifest`
reads.

**Verify** against the paper:

```bash
python scripts/verify_occlusion.py --data-root ../Dataset --benchmarks cuhk rstp icfg
```

For each of the 45 cells, `manifests/occluded/<bench>_<geom>_lNN.json.gz` holds a fixed
sample of 300 gallery images (`numpy.random.default_rng(0).choice(n_gallery, 300)`)
with their seed, the SHA-256 of the occluded image as a raw RGB array **before** JPEG
encoding, the SHA-256 of the JPEG file, the fraction of pixels changed and the image
size. The raw-RGB hash depends only on the source pixels and the occluder code, and
it must match for every image. A JPEG-hash mismatch alone only means a different
libjpeg build inside Pillow. The reference was written with Python 3.10.12,
NumPy 2.2.6 and Pillow 12.3.0, and the check passes for all 45 cells × 300 images
with the code in this repository.

## 2. Measured area

Nominal coverage c is the generator's setting, not the area actually hidden. The
table gives the mean fraction of pixels changed in each cell (from the
`changed_fraction` column of the occlusion manifests, 300 images per cell, in %):

| geometry | c | CUHK-PEDES | ICFG-PEDES | RSTPReid |
|---|---|---|---|---|
| torso | 0.1 | 10.5 | 10.3 | 10.3 |
| torso | 0.2 | 20.4 | 20.2 | 20.2 |
| torso | 0.3 | 30.6 | 30.1 | 30.1 |
| torso | 0.4 | 40.5 | 40.0 | 40.1 |
| torso | 0.5 | 50.5 | 50.0 | 50.0 |
| cutout | 0.1 | 10.5 | 10.3 | 10.3 |
| cutout | 0.2 | 20.9 | 20.4 | 20.4 |
| cutout | 0.3 | 30.1 | 29.7 | 29.5 |
| cutout | 0.4 | 37.3 | 36.8 | 36.6 |
| cutout | 0.5 | 42.3 | 42.0 | 41.5 |
| foliage | 0.1 | 38.2 | 36.9 | 37.5 |
| foliage | 0.2 | 43.4 | 42.8 | 43.1 |
| foliage | 0.3 | 48.5 | 47.2 | 47.6 |
| foliage | 0.4 | 52.0 | 50.2 | 50.8 |
| foliage | 0.5 | 54.7 | 53.2 | 53.9 |

For `foliage`, "changed" includes pixels under semi-transparent leaves, which are
tinted but not fully hidden. This column therefore over-states the destroyed area.
`scripts/measure_occluder_area.py` additionally reports the fully destroyed area,
the fraction of ViT patches touched and the largest clean region per cell.

## 3. Out-of-family (OOF) occluders (evaluation only)

Real object cut-outs from MGCC's occlusion instance library (bench, bike, car, card,
chair, fire hydrant, motorbike, pedestrian, post, road sign, stone, bag, kite,
suitcase, umbrella; 4 PNGs each) are pasted onto gallery images by `sar_person/ogor.py`
(`degrade(..., variant="ogor")`). They are never used in training.

| tag | benchmark | which gallery images | occluder size |
|---|---|---|---|
| `cuhk_ogor_l00_one_per_id` | CUHK-PEDES | the first image of each identity (1,000 of 3,074), as in MGCC's released code | MGCC's per-class area ratios |
| `rstp_ogor_l00_one_per_id` | RSTPReid | first image of each identity (200 of 1,000) | MGCC's per-class ratios |
| `icfg_ogor_l00_one_per_id` | ICFG-PEDES | first image of each identity (1,000 of 19,848) | MGCC's per-class ratios |
| `rstp_ogor_l25`, `_l50`, `_l70` | RSTPReid | all 1,000 | forced to cover c = 0.25 / 0.5 / 0.7 (`--ogor-sweep-scale`) |

Generate with `python tools/run_config.py occlude configs/eval/oof_mgcc.yaml ...`.
`manifests/occluded_oof/<tag>.json.gz` lists the occluded gallery indices and image
IDs. The paper reports `cuhk_ogor_l00_one_per_id` ("OOF CUHK, MGCC protocol") and
`rstp_ogor_l50` ("OOF RSTPReid c = 0.5").

## 4. Occlusion during training (D+A and its variants)

Training uses the same `degrade()` function, applied on the fly in
`sar_person/encoders.py`:

* each training image is occluded with probability p = 0.5 (`--occ-prob 0.5`);
* if occluded, a geometry is drawn uniformly from `--occ-variants` (torso, cutout,
  foliage; foliage only for the aerial Stage 2), and a coverage uniformly from
  `--occ-levels` (0.25, 0.5, 0.7; D+A′ uses 0.25, 0.55, 0.7, none of which is in the
  test grid);
* the random generator is `numpy.random.default_rng((seed, epoch, index))`, so the
  same image meets a different occluder in every epoch, and a run is reproducible
  for a given `--seed`;
* occlusion is applied to the raw training crop first, then the standard
  augmentation (horizontal flip 0.5, pad-10 random crop, random erasing 0.5), which
  draws from an independent stream `default_rng((seed, epoch, index, 7))`.
  Evaluation images are never augmented.

Training occlusion levels {0.25, 0.5, 0.7} and test coverages {0.1, …, 0.5} share
only c = 0.5. The D+A′ control removes even that overlap (Table 1).
