"""Load an official IRRA checkpoint behind this project's encoder interface.

``eval_protocol.evaluate`` only ever asks an encoder for two things --
``encode_images(paths)`` and ``encode_texts(texts)``, both L2-normalised. This
module gives IRRA (Jiang & Ye, CVPR 2023) that interface, so the published
baseline is scored by *the same* script, the same gallery, the same metric code
and the same ground truth as every LoRA run in this repo. That is what makes
the sentence "all self-run baselines use the official public checkpoints
evaluated with our unified protocol" true rather than aspirational.

    python -m sar_person.eval_protocol --benchmark icfg --encoder irra \
        --irra-config <IRRA run>/configs.yaml --out .../icfg_irra

The first thing to do with it is the milestone in EVALUATION_PROTOCOL.md §11:
reproduce IRRA's published R@1 on ICFG within ±1%. If that fails, the protocol
is wrong somewhere and every later number is meaningless -- which is precisely
why this adapter reproduces IRRA's own inference path instead of reusing
``ClipEncoder``'s.

Four things that must match the original, and silently ruin the numbers if not
-----------------------------------------------------------------------------
1. **Preprocessing.** IRRA resizes to ``args.img_size`` -- (384, 128), a 3:1
   portrait -- with no letterboxing. ``ClipEncoder`` letterboxes into a 224x224
   square, which is right for aerial crops of every aspect ratio and wrong here:
   run IRRA through it and R@1 drops several points for no visible reason. This
   module builds IRRA's ``build_transforms(..., is_train=False)``.
2. **Patch stride.** The CUHK/ICFG configs use overlapping patches
   (``stride_size`` 12 for a 16-px patch), which changes the visual token count
   and forces a position-embedding interpolation at build time. It is read from
   configs.yaml and never assumed.
3. **num_classes.** ``IRRA.__init__`` allocates ``classifier`` with one row per
   training identity (11 003 CUHK, 3 102 ICFG, 3 701 RSTP). Guessing makes the
   strict state-dict load fail, so it is read back from the checkpoint's own
   ``classifier.weight`` shape.
4. **fp16.** ``build_model`` calls ``convert_weights``: the released weights are
   half precision and IRRA's own evaluator feeds them float32 tensors, letting
   CLIP cast internally. Both are reproduced. "Fixing" the model to fp32 would
   evaluate a numerically different model than the one that produced the
   published number.

Everything IRRA needs beyond CLIP -- the MLM cross-attention stack, the id
classifier -- is built so the checkpoint loads strictly, then never called:
retrieval uses ``encode_image`` / ``encode_text`` only.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

log = logging.getLogger("sar_person.irra_encoder")

# IRRA's defaults, used only when configs.yaml omits a key.
DEFAULTS = {
    "pretrain_choice": "ViT-B/16",
    "img_size": (384, 128),
    "stride_size": 16,
    "temperature": 0.02,
    "cmt_depth": 4,
    "vocab_size": 49408,
    "loss_names": "sdm+mlm+id",
    "text_length": 77,
    "embed_dim": 512,
}


class Cfg(dict):
    """Attribute-access dict -- what IRRA's code expects from ``edict``.

    Deliberately not ``easydict``: that is an extra dependency for a five-line
    class, and IRRA's model code only ever does ``args.foo``.
    """

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def __setattr__(self, k, v):
        self[k] = v


def add_irra_to_path(irra_root=None) -> Path:
    """Put the IRRA repo root on ``sys.path`` (its imports are absolute).

    ``model``/``datasets``/``utils`` are top-level packages inside that repo, so
    ``from model import build_model`` only resolves with the repo root on the
    path. ``ClipEncoder`` already relies on this being true; here it is made
    explicit and the failure is a sentence instead of an ImportError.
    """
    root = Path(irra_root) if irra_root else \
        Path(__file__).resolve().parents[2] / "IRRA"
    if not (root / "model" / "build.py").is_file():
        raise SystemExit(
            f"IRRA repo not found at {root} (expected model/build.py).\n"
            f"Pass --irra-root /path/to/IRRA.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _yaml_loader():
    """SafeLoader that also understands ``!!python/tuple``.

    IRRA's ``train.py`` writes ``configs.yaml`` with ``yaml.dump`` of an
    argparse Namespace, and ``img_size`` is a tuple there, so the released
    checkpoints ship a config containing::

        img_size: !!python/tuple
        - 384
        - 128

    ``yaml.safe_load`` refuses that tag outright. The alternative,
    ``yaml.unsafe_load``, would execute arbitrary constructors from the file --
    not something to do to a config downloaded off Google Drive. Teaching
    SafeLoader this ONE tag keeps every other unsafe tag rejected.
    """
    import yaml

    class _L(yaml.SafeLoader):
        pass

    _L.add_constructor("tag:yaml.org,2002:python/tuple",
                       lambda loader, node: tuple(
                           loader.construct_sequence(node)))
    return _L


def load_config(config_file, overrides: dict = None) -> Cfg:
    """``configs.yaml`` written by IRRA's training run -> a Cfg for build_model.

    Missing keys are filled from :data:`DEFAULTS` and logged, because a config
    from an older IRRA revision is common and a bare ``AttributeError`` deep
    inside model construction is a miserable way to find that out.
    """
    import yaml

    path = Path(config_file)
    if not path.is_file():
        raise SystemExit(f"IRRA config not found: {path}")
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_yaml_loader()) or {}
    cfg = Cfg(raw)
    filled = []
    for k, v in DEFAULTS.items():
        if k not in cfg:
            cfg[k] = v
            filled.append(k)
    if filled:
        log.warning("[irra] %s missing from %s; using IRRA defaults for: %s",
                    path.name, path.parent.name, ", ".join(filled))
    if isinstance(cfg.get("img_size"), list):
        cfg["img_size"] = tuple(cfg["img_size"])
    cfg["training"] = False
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def find_checkpoint(config_file, ckpt=None) -> Path:
    """The adapter's weights: ``--irra-ckpt``, else ``best.pth`` beside the config."""
    if ckpt:
        p = Path(ckpt)
        if not p.is_file():
            raise SystemExit(f"IRRA checkpoint not found: {p}")
        return p
    here = Path(config_file).parent
    for name in ("best.pth", "best.pth.tar", "checkpoint_best.pth"):
        if (here / name).is_file():
            return here / name
    raise SystemExit(
        f"no checkpoint next to {config_file} (looked for best.pth). "
        f"Pass --irra-ckpt explicitly.")


def state_dict_of(checkpoint) -> dict:
    """The tensor dict, whatever wrapper the release used.

    IRRA's ``Checkpointer`` saves ``{"model": state_dict, "epoch": n, ...}``;
    other releases hand out a bare state_dict or nest it under
    ``state_dict``/``model_state_dict``. DataParallel's ``module.`` prefix is
    stripped, as ``utils.checkpoint.load_state_dict`` does.
    """
    sd = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                sd = checkpoint[key]
                break
    if not isinstance(sd, dict):
        raise SystemExit("unrecognised IRRA checkpoint format: expected a dict")
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):] if k.startswith("module.") else k: v
              for k, v in sd.items()}
    return sd


def infer_num_classes(state: dict, fallback: int = 11003) -> int:
    """Training identity count, read back from the id classifier's shape.

    11 003 (CUHK) / 3 102 (ICFG) / 3 701 (RSTP): building the model with the
    wrong one makes the strict load fail on exactly one tensor, which is a
    confusing way to learn which dataset a checkpoint came from.
    """
    w = state.get("classifier.weight")
    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[0])
    log.warning("[irra] checkpoint has no classifier.weight; assuming "
                "num_classes=%d", fallback)
    return fallback


class IrraEncoder:
    """IRRA behind ``encode_images`` / ``encode_texts``, L2-normalised, fp32 out."""

    name = "irra"

    def __init__(self, config, ckpt=None, irra_root=None, device=None,
                 img_size=None, stride=None):
        import torch

        add_irra_to_path(irra_root)
        from datasets.build import build_transforms
        from model.build import build_model
        from utils.simple_tokenizer import SimpleTokenizer

        self.torch = torch
        self.cfg = load_config(config, {"img_size": img_size,
                                        "stride_size": stride})
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ck_path = find_checkpoint(config, ckpt)
        state = state_dict_of(torch.load(ck_path, map_location="cpu"))
        n_cls = infer_num_classes(state, int(self.cfg.get("num_classes", 11003)))

        model = build_model(self.cfg, num_classes=n_cls)     # fp16, as released
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Loud, because a silent partial load looks like a bad model rather
        # than a bad checkpoint. The MLM head and the id classifier are the
        # only tensors retrieval does not need.
        if missing:
            log.warning("[irra] %d tensor(s) missing from the checkpoint: %s",
                        len(missing), list(missing)[:6])
        if unexpected:
            log.warning("[irra] %d unexpected tensor(s) in the checkpoint: %s",
                        len(unexpected), list(unexpected)[:6])
        if any(k.startswith("base_model.") for k in missing):
            raise SystemExit(
                "the backbone itself did not load (base_model.* tensors are "
                "missing). Check that --irra-config matches --irra-ckpt: a "
                "config with a different pretrain_choice / stride_size builds "
                "a model this checkpoint cannot fill.")
        self.model = model.to(self.device).eval()

        self.transform = build_transforms(img_size=self.cfg.img_size,
                                          is_train=False)
        self.tokenizer = SimpleTokenizer()
        self.text_length = int(self.cfg.get("text_length", 77))
        self.tag = (f"irra:{self.cfg.get('pretrain_choice')}:"
                    f"stride{self.cfg.get('stride_size')}:{Path(config).parent.name}")
        log.info("[irra] %s | img_size=%s stride=%s num_classes=%d | %s",
                 ck_path.name, tuple(self.cfg.img_size),
                 self.cfg.stride_size, n_cls, self.device)

    # ---- encoding ---------------------------------------------------------- #

    def _prep_image(self, path):
        from PIL import Image
        with Image.open(path) as im:
            return self.transform(im.convert("RGB"))

    def encode_images(self, paths: List[str], batch: int = 64) -> np.ndarray:
        t = self.torch
        feats = []
        with t.no_grad():
            for i in range(0, len(paths), batch):
                x = t.stack([self._prep_image(p) for p in paths[i:i + batch]])
                f = self.model.encode_image(x.to(self.device))
                feats.append(f.float().cpu().numpy())
        return _l2(np.concatenate(feats)) if feats else np.zeros((0, 512), "float32")

    def encode_texts(self, texts: List[str], batch: int = 64) -> np.ndarray:
        from datasets.bases import tokenize
        t = self.torch
        feats = []
        with t.no_grad():
            for i in range(0, len(texts), batch):
                toks = t.stack([tokenize(s, self.tokenizer,
                                         text_length=self.text_length,
                                         truncate=True)
                                for s in texts[i:i + batch]])
                f = self.model.encode_text(toks.to(self.device))
                feats.append(f.float().cpu().numpy())
        return _l2(np.concatenate(feats)) if feats else np.zeros((0, 512), "float32")


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


__all__ = ["IrraEncoder", "Cfg", "DEFAULTS", "add_irra_to_path", "load_config",
           "find_checkpoint", "state_dict_of", "infer_num_classes"]
