#!/usr/bin/env python3
"""Convert a published IRRA checkpoint (best.pth) into this repo's full fine-tune
checkpoint format, so train_dataset.py --init-ckpt can continue from it.

    python scripts/convert_irra_ckpt.py \
        ../IRRA/logs/CUHK-PEDES/best.pth ../Dataset/Result/261017/irra_cuhk_init/lora_ckpt.pt

Only the CLIP backbone ("base_model.*") is kept. IRRA's ID classifier, cross-modal
transformer and MLM head are dropped, exactly as D -> D+A dropped D's heads (the
D+A log shows "resumed 301 tensor(s)": backbone only, heads re-initialised).
The script refuses to write unless it finds the same 301 backbone tensors as D's
checkpoint, with identical names and shapes.
"""
import sys
from pathlib import Path
import torch

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
ref = Path(sys.argv[3]) if len(sys.argv) > 3 else \
    Path("../Dataset/Result/260915/isatmp_D_cuhk_author_sdm/lora_ckpt.pt")

ck = torch.load(str(src), map_location="cpu")
sd = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
backbone = {k[len("base_model."):]: v for k, v in sd.items() if k.startswith("base_model.")}
print(f"[irra] {len(sd)} tensors in {src}, {len(backbone)} under base_model.")

r = torch.load(str(ref), map_location="cpu")
rstate = r.get("trainable", r)
want = {k: tuple(v.shape) for k, v in rstate.items() if hasattr(v, "shape")}
got = {k: tuple(v.shape) for k, v in backbone.items() if k in want}
missing = sorted(set(want) - set(got))
badshape = sorted(k for k in got if got[k] != want[k])
if missing or badshape:
    print(f"!! refusing to write: {len(missing)} missing, {len(badshape)} shape mismatches")
    for k in (missing + badshape)[:10]:
        print("   ", k, want.get(k), tuple(backbone[k].shape) if k in backbone else None)
    sys.exit(1)
dst.parent.mkdir(parents=True, exist_ok=True)
blob = {k: v for k, v in r.items() if k != "trainable"} if isinstance(r, dict) else {}
blob.update({"trainable": {k: backbone[k].float().clone() for k in want},
             "epoch": -1, "source": f"converted from IRRA {src}"})
torch.save(blob, str(dst))
print(f"[ok] wrote {len(want)} backbone tensors -> {dst}")
