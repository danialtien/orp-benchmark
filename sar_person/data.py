"""Load (person_id, image, CLIP prompt) triples from the merged Training workbook.

Source of truth:
  - prompts  -> column AD ("Full CLIP Search Prompt", auto-generated) read as value
  - images   -> column AC basename, resolved inside --image-dir
The prompt order may be shuffled by the caller; identity is tracked by `pid` so
the ground-truth match never depends on row/column position.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import openpyxl

SHEET = "Missing Person Profile"
# field columns used by the fallback prompt builder (B..AB minus the auto col)
_FIELDS = ["C","D","F","G","H","I","J","K","L","M","N","O","P","Q","R","S",
           "T","U","V","W","X","Y","Z","AA","AB"]


def _fallback_prompt(ws, r) -> str:
    parts = []
    for c in _FIELDS:
        v = ws[f"{c}{r}"].value
        if v and str(v).strip().lower() not in ("none", "no", ""):
            parts.append(str(v).strip().lower())
    return "a person, " + ", ".join(parts) if parts else "a person"


def load_pairs(xlsx_path: str, image_dir: str, prompt_source: str = "fields") -> List[Dict]:
    """Return list of {pid, label, img_path, prompt, profile}.

    prompt_source:
      "fields" (default) - build the prompt from the row's attributes via
                           PersonProfileSchema.to_prompt() (skips none/no/empty).
      "column"           - use the Excel AD "Full CLIP Search Prompt" value.
    """
    from .profile_schema import PersonProfileSchema
    vwb = openpyxl.load_workbook(xlsx_path, data_only=True)[SHEET]   # cached values
    fwb = openpyxl.load_workbook(xlsx_path)[SHEET]                   # fallback fields
    img_dir = Path(image_dir)
    pairs = []
    pid = 0
    for r in range(5, 200):
        label = vwb[f"A{r}"].value
        if not label or not str(label).strip():
            continue
        profile = PersonProfileSchema.from_excel(vwb, r)
        if prompt_source == "column":
            prompt = vwb[f"AD{r}"].value
            if not prompt or not str(prompt).strip():
                prompt = profile.to_prompt() or _fallback_prompt(fwb, r)
        else:                                   # "fields" (default)
            prompt = profile.to_prompt() or _fallback_prompt(fwb, r)
        ac = vwb[f"AC{r}"].value or fwb[f"AC{r}"].value
        if not ac:
            continue
        name = Path(str(ac).replace("\\", "/")).name
        ip = img_dir / name
        if not ip.exists():
            # try any case-insensitive match
            hits = [p for p in img_dir.iterdir() if p.name.lower() == name.lower()]
            if not hits:
                print(f"[data] WARNING missing image for {label}: {name}")
                continue
            ip = hits[0]
        pairs.append({"pid": pid, "label": str(label).strip(),
                      "img_path": str(ip), "prompt": str(prompt).strip(),
                      "profile": profile})
        pid += 1
    if not pairs:
        raise RuntimeError(f"No (image, prompt) pairs found in {xlsx_path}")
    return pairs


def letterbox(img: Image.Image, size: int = 224, fill=(124, 134, 156)) -> Image.Image:
    """Resize keeping aspect ratio, pad to square `size` (no distortion)."""
    img = img.convert("RGB")
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def mock_image_feat(img_path: str, grid: int = 8) -> np.ndarray:
    """Torch-free image descriptor: per-cell mean RGB over a grid + global hist.
    Used ONLY by the mock encoder for the CPU smoke test."""
    im = letterbox(Image.open(img_path), 64)
    a = np.asarray(im, dtype=np.float32) / 255.0          # 64x64x3
    cells = []
    step = 64 // grid
    for gy in range(grid):
        for gx in range(grid):
            block = a[gy*step:(gy+1)*step, gx*step:(gx+1)*step, :]
            cells.append(block.reshape(-1, 3).mean(0))
    feat = np.concatenate(cells)                           # grid*grid*3
    return feat.astype(np.float32)
