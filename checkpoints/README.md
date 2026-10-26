# Checkpoints

Ten checkpoints are released, one per trained model in the paper. Each is a full
fine-tune of the CLIP ViT-B/16 backbone (149.6 M parameters, fp32, about 600 MB).
The training-only heads (ID classifier, cross-modal MLM transformer) are not in the
files: retrieval uses one cosine score per image-text pair, as in the paper.

| file | paper name | init | seed | selection | SHA-256 (first 16) |
|---|---|---|---|---|---|
| `D.pt` | D | CLIP ViT-B/16 | 0 | best val. epoch 44/60 | `d75f8020dbeb2f3f` |
| `DA_s0.pt` | D+A (reported run) | D | 0 | best val. epoch 4/10 | `1333376db03d43f0` |
| `DA_s1.pt` | D+A | D | 1 | best val. epoch 4/10 | `b9fafbb5ec6042c8` |
| `DA_s2.pt` | D+A | D | 2 | best val. epoch 9/10 | `bd9d4b3738b3422a` |
| `D_plus10.pt` | D+10 (no occlusion) | D | 0 | best val. epoch 4/10 | `ab2153374fa993f4` |
| `DA_prime.pt` | D+A′ (disjoint levels) | D | 0 | best val. epoch 4/10 | `65aaff0302f70428` |
| `IRRA_plus_A.pt` | IRRA+A | IRRA backbone | 0 | best val. epoch 9/10 | `d85eb49b0209f979` |
| `DA_nomad_s0.pt` | D+A→NOMAD | D+A (s0) | 0 | last epoch | `a22f0a89c73b4f0f` |
| `DA_nomad_s1.pt` | D+A→NOMAD | D+A (s0) | 1 | last epoch | `2ce899247feb6668` |
| `DA_nomad_s2.pt` | D+A→NOMAD | D+A (s0) | 2 | last epoch | `804ae4b68af67bba` |

Epochs are 0-based. The full hashes, the exact training arguments and the original
run each file comes from are in `configs/train/<name>.yaml`. "Best val." means the
highest text-to-image Rank-1 on 300 identities held out from the CUHK-PEDES
**training** split; no checkpoint was ever chosen on a test set. The aerial models
use the final epoch, so NOMAD-val plays no part in choosing them.

`irra_init` (the published IRRA CUHK-PEDES weights converted to this format, the
start point of IRRA+A) is not redistributed. Download IRRA's `best.pth` from
<https://github.com/anosorae/IRRA> and convert it (see `configs/train/irra_init.yaml`).

## Download

```bash
python scripts/fetch_checkpoints.py              # all ten, verified against the configs
python scripts/fetch_checkpoints.py D DA_s0      # a subset
python scripts/fetch_checkpoints.py --verify-only
```

**Location:** `TO BE FILLED AT PUBLICATION` (Zenodo record / Hugging Face model repo).
During double-blind review the files are shared through an anonymous link given in
the response to reviewers. Set `BASE_URL` in `scripts/fetch_checkpoints.py` to the
final location.

## Using a checkpoint

```bash
python -m sar_person.eval_protocol --benchmark cuhk --data-root ../Dataset \
    --encoder clip --pretrain ViT-B/16 --img-size 384 128 --full-finetune \
    --ckpt checkpoints/DA_s0.pt --batch 64 --out runs/DA_s0/eval/cuhk_clean
```

or `python tools/run_config.py eval configs/eval/orp_ground.yaml --model DA_s0 ...`,
which finds `checkpoints/DA_s0.pt` by itself.

## For maintainers

`tools/package_checkpoints.py --result-root <private Result tree> --out <upload dir>`
copies the ten files, names them as above and writes `SHA256SUMS`. Run it with
`--check-only` on the training server first: it confirms that the file carrying each
recorded hash is the one the evaluation logs point to.
