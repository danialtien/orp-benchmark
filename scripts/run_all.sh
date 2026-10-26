#!/usr/bin/env bash
# Reproduce the paper end to end. Each stage can be run on its own:
#
#   bash scripts/run_all.sh check      # data + occlusion verification (no GPU)
#   bash scripts/run_all.sh occlude    # build the occluded galleries (no GPU)
#   bash scripts/run_all.sh eval       # score the RELEASED checkpoints (1 GPU)
#   bash scripts/run_all.sh train      # retrain every model (1 GPU, ~20 GPU-hours)
#   bash scripts/run_all.sh analyse    # tables, CIs, seed checks (CPU, seconds)
#
# Variables: DATA=../Dataset  OUT=runs  GPU=0
# 'eval' scores checkpoints/<name>.pt, or runs/<name>/... if you retrained.
# After 'eval', compare your scores with the paper's, cell by cell:
#   python tools/compare_runs.py --out-root runs
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-../Dataset}; OUT=${OUT:-runs}; GPU=${GPU:-0}
RC="python tools/run_config.py"

check() {
  python scripts/verify_evalsets.py      --data-root "$DATA" --benchmarks cuhk icfg rstp nomad
  python scripts/verify_training_data.py --data-root "$DATA"
  python scripts/verify_occlusion.py     --data-root "$DATA" --benchmarks cuhk rstp icfg
}
occlude() {
  $RC occlude configs/eval/orp_ground.yaml --data-root "$DATA" --out-root "$OUT"
  $RC occlude configs/eval/oof_mgcc.yaml   --data-root "$DATA" --out-root "$OUT"
}
evaluate() {
  for m in clip_zs D D_plus10 DA_s0 DA_s1 DA_s2 DA_prime IRRA_plus_A; do
    $RC eval configs/eval/orp_ground.yaml --model "$m" --data-root "$DATA" --out-root "$OUT" --gpu "$GPU"
  done
  for m in D DA_s0; do
    $RC eval configs/eval/oof_mgcc.yaml --model "$m" --data-root "$DATA" --out-root "$OUT" --gpu "$GPU"
  done
  for m in clip_zs DA_s0 DA_nomad_s0 DA_nomad_s1 DA_nomad_s2; do
    $RC eval configs/eval/nomad_val.yaml --model "$m" --data-root "$DATA" --out-root "$OUT" --gpu "$GPU"
  done
  for m in DA_nomad_s0 DA_nomad_s1 DA_nomad_s2; do
    $RC eval configs/eval/orp_ground.yaml --model "$m" --benchmarks cuhk rstp --clean-only \
        --data-root "$DATA" --out-root "$OUT" --gpu "$GPU"
  done
  echo "published IRRA: see configs/eval/orp_ground.yaml (irra_published) -- needs a clone of IRRA"
}
train() {
  # order matters: D first, then everything initialised from it
  for c in D DA_s0 DA_s1 DA_s2 D_plus10 DA_prime DA_nomad_s0 DA_nomad_s1 DA_nomad_s2; do
    $RC train "configs/train/$c.yaml" --data-root "$DATA" --out-root "$OUT" --gpu "$GPU"
  done
  echo "IRRA+A: first  $RC train configs/train/irra_init.yaml --out-root $OUT --irra-pth <IRRA>/logs/CUHK-PEDES/best.pth"
  echo "        then   $RC train configs/train/IRRA_plus_A.yaml --data-root $DATA --out-root $OUT --gpu $GPU"
}
analyse() {
  python analysis/reproduce_paper.py
  python analysis/seed_consistency.py
  python analysis/ci_stability.py
}
case "${1:-}" in
  check) check ;; occlude) occlude ;; eval) evaluate ;; train) train ;; analyse) analyse ;;
  *) sed -n 2,14p "$0"; exit 1 ;;
esac
