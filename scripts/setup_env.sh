#!/usr/bin/env bash
# Create a virtual environment for training / evaluation.
#   bash scripts/setup_env.sh                                   # CUDA 12.4 wheels
#   TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/setup_env.sh
#   TORCH_INDEX=cpu bash scripts/setup_env.sh                    # CPU only (analysis, data checks)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
if [ "$TORCH_INDEX" = "cpu" ]; then
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch torchvision --index-url "$TORCH_INDEX"
fi
pip install -r requirements.txt
python - <<'PY'
import torch, numpy, PIL
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("numpy", numpy.__version__, "| Pillow", PIL.__version__)
PY
python -m pytest -q tests || echo "NOTE: some tests need the datasets or a GPU; see README"
echo "done. Activate with: source .venv/bin/activate"
