"""Score an official MGCC checkpoint through this project's eval protocol.

MGCC (Wu et al., AAAI-24) is the closest published work to ours: same task, same
CLIP backbone family, occlusion as the headline problem. Its published table is
in units we cannot use directly -- R@K/mAP/Rsum on *their* Occluded-* datasets,
no mINP -- so the only way to put MGCC in a table next to us is to run it, on our
gallery, through ``eval_protocol``. This adapter does that.

    python -m sar_person.eval_protocol \
        --manifest .../icfg_ogor_l00_one_per_id.json \
        --encoder mgcc --mgcc-root ../MGCC --mgcc-ckpt .../best.pth.tar \
        --out ../Dataset/Result/tier2/icfg_ogor_mgcc

Milestone before believing any of it: reproduce MGCC's published R@1 on
Occluded-ICFG (59.28) within +-1. If it does not land, the difference is in the
dataset build or the preprocessing, not in the metric code -- see MGCC's two
non-obvious protocol choices below.

Why this is NOT an ``encode_images``/``encode_texts`` adapter
------------------------------------------------------------
IRRA is dual-tower: embed the gallery once, embed the queries once, dot product.
MGCC is not. Its score is the mean of four terms (``model.get_similarity``):
image-text, image-word, patch-text and patch-word, and the last three are
*attention-weighted aggregations over a patch x word similarity matrix* with
learned mixing matrices. There is no pair of embedding tables whose dot product
equals that, so this class exposes ``similarity()`` instead and
``eval_protocol.evaluate`` routes around the feature cache.

That has a cost worth knowing before launching a run: pairwise scoring is
O(Nq x Ng) forward-ish work rather than O(Nq + Ng). Patch and word features are
still extracted once each (cheap); the aggregation is then a chunked matmul over
[chunk_q, n_patch, chunk_g, n_word], which is where both the time and the memory
go. ``--mgcc-chunk`` trades one against the other.

Two MGCC protocol choices that will silently shift your numbers
---------------------------------------------------------------
1. **ICFG's test split is truncated.** ``process_data.py`` keeps only the first
   10 000 test rows (``len(test_json) < 10000``) and pushes the remainder into
   val. Standard ICFG test is 19 848 images. A smaller gallery raises R@1, so
   MGCC's 59.28 is not on the gallery the rest of the field reports. Build the
   matching EvalSet, or say in the caption which gallery you used.
2. **CUHK drops three identities.** ids 1369, 4116 and 6116 have a single image
   each and are removed, with the remaining ids re-indexed.

Neither is wrong -- both are undocumented in the paper.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

log = logging.getLogger("sar_person.mgcc_encoder")

#: MGCC's released configs. Overridable via ``opts``; anything not listed is
#: read from the checkpoint or from MGCC's own ``options.py`` defaults.
DEFAULTS = {
    "img_model": "CLIP",
    "txt_model": "CLIP",
    "feature_length": 512,
    "max_words": 100,
    "max_patch": 192,
    "Rv": 0.3,                 # image token-selection ratio  (rho_i)
    "Rt": 0.4,                 # text  token-selection ratio  (rho_t)
    "Topk_Selection": True,
    "Aggregation_Type": "Attention",
    "img_text_logits": True,
    "img_word_logits": True,
    "patch_text_logits": True,
    "patch_word_logits": True,
    "softmax_t": 0.01,
    "mode": "test",
}


class MgccEncoder:
    """Wraps ``TextImgPersonReidNet`` behind a ``similarity(paths, texts)`` call.

    ``mgcc_root`` is a clone of https://github.com/littlexinyi/MGCC; its ``src``
    directory is put on ``sys.path`` because MGCC's modules import each other as
    top-level packages (``from model.image_clip import ...``), so they only
    resolve from inside ``src``.
    """

    def __init__(self, mgcc_root: str | Path, ckpt: str | Path,
                 device: str | None = None, chunk: int = 256,
                 img_size: Sequence[int] = (384, 128), opts: dict | None = None):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk = int(chunk)

        root = Path(mgcc_root).expanduser().resolve()
        src = root / "src"
        if not (src / "model" / "model.py").is_file():
            raise FileNotFoundError(
                f"{src}/model/model.py not found. Clone MGCC first:\n"
                f"    git clone https://github.com/littlexinyi/MGCC {root}")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        opt = _Opt({**DEFAULTS, **(opts or {})})
        opt.batch_size = self.chunk
        opt.max_token_length = opt.max_words
        opt.img_size = tuple(img_size)

        from model.model import TextImgPersonReidNet          # noqa: E402

        state = torch.load(str(ckpt), map_location="cpu")
        state = state.get("network", state.get("state_dict", state))
        state = {k[7:] if k.startswith("module.") else k: v
                 for k, v in state.items()}

        self.model = TextImgPersonReidNet(opt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            # Not fatal by construction -- MGCC allocates loss-side heads that
            # retrieval never touches -- but a *large* miss means the config
            # here does not describe the checkpoint, and the numbers would be
            # from a partly random model. Fail loudly on that.
            log.warning("[mgcc] %d missing key(s), first: %s",
                        len(missing), missing[:5])
            if len(missing) > 0.1 * len(state):
                raise RuntimeError(
                    f"{len(missing)}/{len(state)} weights missing -- the opts "
                    f"here do not match this checkpoint. Compare with the "
                    f"experiment/*/test.sh the checkpoint was produced by.")
        if unexpected:
            log.info("[mgcc] %d unexpected key(s) ignored", len(unexpected))

        self.model.to(self.device).eval()
        self.opt = opt
        self._tok = _tokenizer(root)

    # -- feature extraction ------------------------------------------------
    def _images(self, paths: Sequence[str], batch: int = 32):
        """-> (global [N,d], patch [N,P,d]), both L2-normalised as MGCC does."""
        from PIL import Image
        import torchvision.transforms as T
        t = self.torch
        tf = T.Compose([
            T.Resize(self.opt.img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711)),
        ])
        gs, ps = [], []
        with t.no_grad():
            for i in range(0, len(paths), batch):
                x = t.stack([tf(Image.open(p).convert("RGB"))
                             for p in paths[i:i + batch]]).to(self.device)
                out = self.model.img_embedding(x)
                g, p = out[0], out[1]                 # test mode returns 3
                gs.append(g.float().cpu())
                ps.append(p.float().cpu())
        return t.cat(gs), t.cat(ps)

    def _texts(self, texts: Sequence[str], batch: int = 64):
        t = self.torch
        gs, ws = [], []
        with t.no_grad():
            for i in range(0, len(texts), batch):
                tok = self._tok(list(texts[i:i + batch])).to(self.device)
                out = self.model.txt_embedding(tok)
                g, w = out[0], out[1]
                gs.append(g.float().cpu())
                ws.append(w.float().cpu())
        return t.cat(gs), t.cat(ws)

    # -- the interface eval_protocol looks for -----------------------------
    def similarity(self, gallery_paths: Sequence[str],
                   query_texts: Sequence[str]) -> np.ndarray:
        """[n_query, n_gallery] float32, MGCC's own four-term fused score."""
        t = self.torch
        log.info("[mgcc] encoding %d gallery image(s)", len(gallery_paths))
        gi, gp = self._images(list(gallery_paths))
        log.info("[mgcc] encoding %d query text(s)", len(query_texts))
        tg, tw = self._texts(list(query_texts))

        nq, ng = len(query_texts), len(gallery_paths)
        S = np.empty((nq, ng), dtype=np.float32)
        c = self.chunk
        with t.no_grad():
            for qi in range(0, nq, c):
                q_g = tg[qi:qi + c].to(self.device)
                q_w = tw[qi:qi + c].to(self.device)
                for gi_ in range(0, ng, c):
                    i_g = gi[gi_:gi_ + c].to(self.device)
                    i_p = gp[gi_:gi_ + c].to(self.device)
                    # get_similarity returns [n_img, n_txt]; we want [q, g]
                    s = self.model.get_similarity(i_g, i_p, q_g, q_w)
                    S[qi:qi + c, gi_:gi_ + c] = s.t().float().cpu().numpy()
        return S


class _Opt:
    """Attribute-access options object -- what MGCC's modules expect."""

    def __init__(self, d: dict):
        self.__dict__.update(d)

    def __getattr__(self, k):                                  # pragma: no cover
        raise AttributeError(
            f"MGCC asked for option {k!r}, which this adapter does not set. "
            f"Add it via --mgcc-opts '{k}=...' after checking what the "
            f"matching experiment/*/train.sh used.")


def _tokenizer(root: Path):
    """CLIP BPE tokenizer, from MGCC's vendored copy if present."""
    try:
        from clip.simple_tokenizer import SimpleTokenizer
        import clip
        return lambda xs: clip.tokenize(xs, truncate=True)
    except Exception:                                          # noqa: BLE001
        pass
    try:
        from transformers import CLIPTokenizerFast
        tk = CLIPTokenizerFast.from_pretrained(
            str(root / "src" / "pretrain" / "clip-vit-base-patch32"))
        import torch

        def fn(xs):
            o = tk(xs, padding="max_length", truncation=True,
                   max_length=77, return_tensors="pt")
            return o["input_ids"]
        return fn
    except Exception as e:                                     # noqa: BLE001
        raise ImportError(
            "no CLIP tokenizer available. Install openai/CLIP "
            "(pip install git+https://github.com/openai/CLIP.git) or "
            "transformers, and make sure MGCC's pretrain/ folder is "
            "populated per its README.") from e


def get_encoder(mgcc_root, ckpt, device=None, chunk=256, img_size=(384, 128),
                opts=None) -> MgccEncoder:
    return MgccEncoder(mgcc_root, ckpt, device=device, chunk=chunk,
                       img_size=img_size, opts=opts)
