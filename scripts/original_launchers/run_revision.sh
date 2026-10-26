#!/usr/bin/env bash
# =============================================================================
# run_revision.sh -- GPU runs requested by the 2026-10-17 internal review of the
# ORP paper. Run from the sar_person repo root, inside its venv:
#
#   GPU=0 bash scripts/revision_261017/run_revision.sh ctrl      # D + 10 ep, NO occlusion
#   GPU=1 bash scripts/revision_261017/run_revision.sh irra      # D+A recipe from IRRA weights
#   GPU=0 bash scripts/revision_261017/run_revision.sh seed 1    # D+A, seed 1 (also: seed 2)
#   GPU=1 bash scripts/revision_261017/run_revision.sh disjoint  # D+A, train levels 0.25 0.55 0.7
#   GPU=0 bash scripts/revision_261017/run_revision.sh eval <run_name> [benchmarks]
#
#   Overnight, two GPUs:
#     GPU=0 bash .../run_revision.sh chain0   # ctrl -> eval -> seed 1 -> eval
#     GPU=1 bash .../run_revision.sh chain1   # irra -> eval -> disjoint -> eval
#
# Every training run uses the D+A recipe verbatim (read off the D+A log header,
# Result/260916/logs/isatmp_Docc_cuhk_author_sdm.log) and changes ONE thing.
# After start-up the script diffs the new log header against the D+A header and
# KILLS the run if anything other than the intended change differs.
# One run needs ~13 GB: never two runs on the same 3090.
# =============================================================================
set -uo pipefail
GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-../Dataset}
OUT_ROOT=${OUT_ROOT:-$DATA_ROOT/Result/261017}
CACHE=${CACHE:-$DATA_ROOT/Result/evalsets}
OCC_ROOT=${OCC_ROOT:-$DATA_ROOT/Result/occluded}
D_CKPT=${D_CKPT:-$DATA_ROOT/Result/260915/isatmp_D_cuhk_author_sdm/lora_ckpt.pt}
IRRA_PTH=${IRRA_PTH:-../IRRA/logs/CUHK-PEDES/best.pth}
REF_LOG=${REF_LOG:-$DATA_ROOT/Result/260916/logs/isatmp_Docc_cuhk_author_sdm.log}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_ROOT/logs"

# --- the D+A recipe ----------------------------------------------------------
RECIPE=(--datasets cuhk_author --data-root "$DATA_ROOT" --limit 0
        --encoder clip --pretrain ViT-B/16 --img-size 384 128 --full-finetune
        --loss sdm --sdm-temp 0.02 --hard-neg-weight 0
        --id-loss-weight 1.0 --mlm-loss-weight 1.0 --aug
        --sampler random --optimizer adam --lr 1e-5 --lr-factor 5.0 --bias-lr-factor 2.0
        --weight-decay 4e-5 --weight-decay-bias 0
        --lr-schedule constant --warmup-epochs 5 --epochs 10
        --eval-every 5 --select-metric t2i_R@1 --batch 64)
OCC_ON=(--occ-prob 0.5 --occ-variants torso cutout foliage)

header() {  # normalised [clip]/optim lines of a training log
  grep -E '^\[clip\] (loss=|aux heads|optim:|train: )' "$1" \
    | grep -v 'occlusion aug' | sed -E 's#/[^ ]+##g'
}

vram_ok() {
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -1)
  [ "$free" -ge 14000 ] || { echo "!! GPU $GPU has only ${free} MiB free"; return 1; }
}

train() {   # train <name> <seed> <extra args...>
  local name=$1 seed=$2; shift 2
  local out="$OUT_ROOT/$name" log="$OUT_ROOT/logs/$name.log"
  [ -s "$out/lora_ckpt.pt" ] && { echo "[skip] $name already trained"; return 0; }
  vram_ok || return 1
  echo "[train] $name on GPU $GPU -> $log"
  # -u / PYTHONUNBUFFERED: the [clip] header lines are print()s on stdout, which
  # Python block-buffers when redirected to a file. Without this they do not reach
  # the log for a long time, and the parity gate below sees an empty header.
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 python -u -m sar_person.train_dataset \
      "${RECIPE[@]}" "$@" --seed "$seed" --gpu "$GPU" --out "$out" > "$log" 2>&1 &
  local pid=$!
  # parity gate: wait for the header, diff it against D+A, kill only on a REAL mismatch
  local seen=0
  for _ in $(seq 1 180); do
    grep -q 'image aug' "$log" 2>/dev/null && { seen=1; break; }
    kill -0 $pid 2>/dev/null || break; sleep 10
  done
  if ! kill -0 $pid 2>/dev/null; then
    echo "!! $name exited during start-up; last lines of $log:"; tail -n 30 "$log"; return 1
  fi
  if [ $seen -eq 0 ]; then
    echo "!! $name: header not seen after 30 min; NOT killing. Check $log by hand."
  elif ! diff <(header "$REF_LOG") <(header "$log") > "$OUT_ROOT/logs/$name.parity.diff"; then
    echo "!! $name differs from the D+A recipe (see $OUT_ROOT/logs/$name.parity.diff); killing."
    cat "$OUT_ROOT/logs/$name.parity.diff"; kill $pid; return 1
  else
    echo "[parity] $name matches the D+A recipe"
  fi
  grep 'occlusion aug' "$log" || echo "[train] $name: occlusion augmentation OFF"
  wait $pid; local rc=$?
  echo "[train] $name finished rc=$rc"; return $rc
}

evaluate() {  # evaluate <name> [benchmarks]
  local name=$1 benches=${2:-"cuhk rstp icfg"} ck="$OUT_ROOT/$1/lora_ckpt.pt"
  [ -s "$ck" ] || { echo "!! no checkpoint $ck"; return 1; }
  for B in $benches; do
    local o="$OUT_ROOT/$name/${B}_clean"
    [ -s "$o/metrics.json" ] || CUDA_VISIBLE_DEVICES="$GPU" python -m sar_person.eval_protocol \
        --benchmark "$B" --data-root "$DATA_ROOT" --evalset-cache "$CACHE" \
        --encoder clip --pretrain ViT-B/16 --ckpt "$ck" --full-finetune \
        --img-size 384 128 --gpu "$GPU" --batch 64 --tag "$name" --out "$o" \
        >> "$OUT_ROOT/logs/$name.eval.log" 2>&1 || echo "!! eval failed: $name/$B clean"
    for V in torso cutout foliage; do for P in 10 20 30 40 50; do
      o="$OUT_ROOT/$name/occ/${B}_${V}_l${P}"
      [ -s "$o/metrics.json" ] && continue
      CUDA_VISIBLE_DEVICES="$GPU" python -m sar_person.eval_protocol \
        --manifest "$OCC_ROOT/manifests/${B}_${V}_l${P}.json" --data-root "$DATA_ROOT" \
        --encoder clip --pretrain ViT-B/16 --ckpt "$ck" --full-finetune \
        --img-size 384 128 --gpu "$GPU" --batch 64 --tag "$name" --out "$o" \
        >> "$OUT_ROOT/logs/$name.eval.log" 2>&1 || echo "!! eval failed: $name/$B $V $P"
    done; done
  done
  echo "[eval] $name done ($benches)"
}

irra_init() {
  local init="$OUT_ROOT/irra_cuhk_init/lora_ckpt.pt"
  [ -s "$init" ] || python "$HERE/convert_irra_ckpt.py" "$IRRA_PTH" "$init" "$D_CKPT" || return 1
  # gate: the converted weights must reproduce IRRA's clean CUHK score (73.16 R@1)
  local o="$OUT_ROOT/irra_cuhk_init/cuhk_clean"
  [ -s "$o/metrics.json" ] || CUDA_VISIBLE_DEVICES="$GPU" python -m sar_person.eval_protocol \
      --benchmark cuhk --data-root "$DATA_ROOT" --evalset-cache "$CACHE" \
      --encoder clip --pretrain ViT-B/16 --ckpt "$init" --full-finetune \
      --img-size 384 128 --gpu "$GPU" --batch 64 --tag irra_cuhk_init --out "$o" \
      >> "$OUT_ROOT/logs/irra_cuhk_init.eval.log" 2>&1
  python - "$o/metrics.json" <<'PY' || return 1
import json, sys
r1 = 100 * json.load(open(sys.argv[1]))["overall"]["R@1"]
print(f"[gate] converted IRRA weights: CUHK clean R@1 = {r1:.2f} (IRRA encoder: 73.16)")
sys.exit(0 if r1 > 72.0 else 1)
PY
}

case "${1:-}" in
  ctrl)     train ctrl_D_plus10_noocc 0 --init-ckpt "$D_CKPT" ;;
  irra)     irra_init && train irra_plus_occ 0 --init-ckpt "$OUT_ROOT/irra_cuhk_init/lora_ckpt.pt" \
                "${OCC_ON[@]}" --occ-levels 0.25 0.5 0.7 ;;
  seed)     s=${2:?seed}; train "Docc_seed$s" "$s" --init-ckpt "$D_CKPT" "${OCC_ON[@]}" --occ-levels 0.25 0.5 0.7 ;;
  disjoint) train Docc_lv_025_055_07 0 --init-ckpt "$D_CKPT" "${OCC_ON[@]}" --occ-levels 0.25 0.55 0.7 ;;
  eval)     evaluate "${2:?run name}" "${3:-cuhk rstp icfg}" ;;
  chain0)   "$0" ctrl && "$0" eval ctrl_D_plus10_noocc && "$0" seed 1 && "$0" eval Docc_seed1 "cuhk rstp" \
              && "$0" seed 2 && "$0" eval Docc_seed2 "cuhk rstp" ;;
  chain1)   "$0" irra && "$0" eval irra_plus_occ && "$0" disjoint && "$0" eval Docc_lv_025_055_07 "cuhk rstp" ;;
  *) sed -n 2,22p "$0"; exit 1 ;;
esac
