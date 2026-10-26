"""Encoders that map images and prompts into one shared, L2-normalised space.

- ClipEncoder : real CLIP ViT-B/16 from IRRA's backbone (+ optional LoRA via
                IRRA model/petl.py). Needs torch. Use on the GPU box.
- MockEncoder : torch-free stand-in (random projections of cheap features) so the
                full pipeline (data -> similarity -> metrics -> plots) can be
                smoke-tested on CPU and produce sample artifacts. NOT a real model.
"""
from __future__ import annotations
import logging
import sys, types
from pathlib import Path
from typing import List
import numpy as np

IRRA_ROOT = Path(__file__).resolve().parents[1]
if str(IRRA_ROOT) not in sys.path:
    sys.path.insert(0, str(IRRA_ROOT))

from .data import mock_image_feat, letterbox


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


# --------------------------------------------------------------------------- #
# Mock encoder (CPU, no torch)                                                 #
# --------------------------------------------------------------------------- #
class MockEncoder:
    name = "mock"

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.vocab = {}
        self.Wi = None
        self.Wt = None

    def _text_bow(self, texts: List[str]) -> np.ndarray:
        import re
        toks = [re.findall(r"[a-z0-9]+", t.lower()) for t in texts]
        if not self.vocab:
            for ts in toks:
                for w in ts:
                    self.vocab.setdefault(w, len(self.vocab))
        V = len(self.vocab)
        M = np.zeros((len(texts), V), np.float32)
        for i, ts in enumerate(toks):
            for w in ts:
                j = self.vocab.get(w)
                if j is not None:
                    M[i, j] += 1.0
        return M

    def encode_images(self, paths: List[str]) -> np.ndarray:
        F = np.stack([mock_image_feat(p) for p in paths])
        if self.Wi is None:
            self.Wi = self.rng.standard_normal((F.shape[1], self.dim)).astype(np.float32)
        return l2norm(F @ self.Wi)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        B = self._text_bow(texts)
        if self.Wt is None:
            self.Wt = self.rng.standard_normal((B.shape[1], self.dim)).astype(np.float32)
        return l2norm(B @ self.Wt)

    def simulate_finetune(self, img_emb, txt_emb, gt_col, alpha=0.85):
        """CPU stand-in for training: pull each prompt toward its paired image.
        Returns a NEW text-embedding matrix (same indexing as txt_emb)."""
        out = txt_emb.copy()
        for row, col in enumerate(gt_col):
            out[col] = alpha * img_emb[row] + (1 - alpha) * txt_emb[col]
        return l2norm(out)


# --------------------------------------------------------------------------- #
# Real CLIP encoder (torch + IRRA)                                            #
# --------------------------------------------------------------------------- #
def infer_lora_rank(state_or_path) -> int | None:
    """Read the LoRA rank straight off the checkpoint tensors.

    ``LoRALinear`` stores ``A`` of shape ``(r, in_features)``, so the rank is in
    the file and never needs to be guessed from a folder name. It is not stored
    as metadata anywhere -- ``run.py`` never wrote it -- so this is the only
    reliable source.

    (``alpha`` genuinely is unrecoverable: it only ever scales the forward pass,
    leaving no trace in the weights. It must come from the run's metrics.json or
    from knowing the default. That asymmetry is why rank is inferred and alpha
    is asked for.)
    """
    state = state_or_path
    if isinstance(state_or_path, (str, Path)):
        import torch
        ck = torch.load(str(state_or_path), map_location="cpu")
        state = ck.get("trainable", ck) if isinstance(ck, dict) else ck
    ranks = {tuple(v.shape)[0] for k, v in state.items()
             if k.endswith(".A") and hasattr(v, "shape") and len(v.shape) == 2}
    if len(ranks) == 1:
        return int(next(iter(ranks)))
    if len(ranks) > 1:
        log = logging.getLogger(__name__)
        log.warning("[clip] checkpoint has mixed LoRA ranks %s", sorted(ranks))
    return None


def is_full_finetune_ckpt(state_or_path) -> bool:
    """True when the checkpoint holds a FULL backbone, not a LoRA adapter.

    ``train_dataset --full-finetune`` saves ``named_parameters()`` of the
    untouched CLIP model: plain keys like ``visual.conv1.weight``, no ``.A`` /
    ``.B``, and no ``.lin.`` infix. A LoRA checkpoint has the opposite shape --
    it is small, and every trained linear lives under ``<name>.lin``.

    This matters because :meth:`ClipEncoder.load_trainable` injects LoRA before
    copying tensors, which *renames* every wrapped linear
    (``...c_fc.weight`` -> ``...c_fc.lin.weight``). Feed it a full fine-tune
    checkpoint and all ~200 keys become "unknown key", strict loading raises,
    and a lenient loader would hand back zero-shot CLIP with a straight face.
    Detecting the kind from the tensors -- never from a folder name or a flag
    the caller forgot -- is the only version that cannot silently mislead.
    """
    state = state_or_path
    if isinstance(state_or_path, (str, Path)):
        import torch
        # mmap avoids materialising ~600 MB of fp32 weights just to read the
        # key names; eval_protocol calls this once per run and the checkpoint
        # is then loaded again for real. Not every torch build supports it.
        try:
            ck = torch.load(str(state_or_path), map_location="cpu", mmap=True)
        except (TypeError, RuntimeError, ValueError):
            ck = torch.load(str(state_or_path), map_location="cpu")
        state = ck.get("trainable", ck) if isinstance(ck, dict) else ck
    keys = list(state.keys())
    if not keys:
        return False
    if any(k.endswith(".A") or k.endswith(".B") for k in keys):
        return False                      # LoRA tensors present -> adapter
    # A full-FT save carries the patch-embedding conv and the token embedding;
    # a LoRA save carries neither (those stay frozen and are never written).
    return any(k.endswith("visual.conv1.weight") for k in keys) or len(keys) > 150


class ClipEncoder:
    name = "clip"

    def __init__(self, pretrain="ViT-B/16", img_size=(224, 224), stride=16, device=None):
        import torch
        from model.clip_model import build_CLIP_from_openai_pretrained
        from utils.simple_tokenizer import SimpleTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = build_CLIP_from_openai_pretrained(pretrain, img_size, stride)
        self.model = self.model.float().to(self.device)
        self.tokenizer = SimpleTokenizer()
        hw = tuple(img_size) if isinstance(img_size, (tuple, list)) else \
            (img_size, img_size)
        self.img_hw = (int(hw[0]), int(hw[-1]))          # (height, width)
        # Kept for callers that expect a scalar; only meaningful when square.
        self.img_size = self.img_hw[0]
        # Square -> letterbox (pad to preserve aspect): right for aerial crops,
        # whose aspect ratio varies with viewing angle and where stretching a
        # top-down figure into a square destroys the body proportions the
        # captions describe.
        # Non-square -> plain resize, no padding. This is what IRRA/TBPS-CLIP/
        # RDE do at 384x128, and matching them is the point of the option: a
        # letterboxed 384x128 would waste most of the frame on grey bars,
        # because person crops are already ~3:1.
        self.letterbox_pad = self.img_hw[0] == self.img_hw[1]
        self._mean = (0.48145466, 0.4578275, 0.40821073)
        self._std = (0.26862954, 0.26130258, 0.27577711)

    # ---- preprocessing ----
    def _prep_image(self, path, aug=None, train_rng=None):
        """``aug`` is an optional PIL->PIL transform applied *before* letterbox.

        Before, not after: occlusion augmentation has to see the crop at its
        native aspect ratio, otherwise "cover the lower 45%" means the lower 45%
        of a letterboxed square -- part of which is grey padding, not person.

        ``train_rng`` is a numpy Generator. Pass one ONLY on the training path:
        it turns on flip + pad-crop + random erasing, and a random transform on
        the evaluation path would randomise the gallery itself. ``encode_images``
        never passes it, which is the whole reason this is a parameter and not a
        flag on the object.
        """
        import numpy as np
        PILImage = __import__("PIL.Image", fromlist=["Image"])
        im = PILImage.open(path)
        if aug is not None:
            im = aug(im.convert("RGB"))
        if self.letterbox_pad:
            im = letterbox(im, self.img_size)
        else:
            h, w = self.img_hw
            im = im.convert("RGB").resize((w, h), PILImage.BICUBIC)
        if train_rng is not None:
            im = _geom_aug(im, train_rng)
        a = (np.asarray(im, np.float32) / 255.0 - self._mean) / self._std
        if train_rng is not None:
            a = _random_erase(a, train_rng)
        return self.torch.tensor(a).permute(2, 0, 1)

    def _tok(self, text):
        from datasets.bases import tokenize
        return tokenize(text, self.tokenizer, text_length=77, truncate=True)

    def _pool_img(self, f):
        return f[:, 0, :] if f.dim() == 3 else f          # CLS token

    def _pool_txt(self, f, toks):
        if f.dim() == 3:                                  # EOT token (= max id)
            idx = toks.argmax(dim=-1)
            f = f[self.torch.arange(f.shape[0], device=f.device), idx]
        return f

    # ---- encoding ----
    def encode_images(self, paths, batch=32):
        t = self.torch
        feats = []
        self.model.eval()
        with t.no_grad():
            for i in range(0, len(paths), batch):
                x = t.stack([self._prep_image(p) for p in paths[i:i+batch]]).to(self.device)
                f = self._pool_img(self.model.encode_image(x).float())
                feats.append(f.cpu().numpy())
        return l2norm(np.concatenate(feats))

    def encode_texts(self, texts, batch=32):
        t = self.torch
        feats = []
        self.model.eval()
        with t.no_grad():
            for i in range(0, len(texts), batch):
                toks = t.stack([self._tok(x) for x in texts[i:i+batch]]).to(self.device)
                f = self._pool_txt(self.model.encode_text(toks).float(), toks)
                feats.append(f.cpu().numpy())
        return l2norm(np.concatenate(feats))

    # ---- LoRA injection (shared by train_lora and load_trainable) ----
    def _inject_lora(self, lora_rank=16, lora_alpha=16, attn=False):
        """Wrap the transformer MLP linears with LoRA and mark LayerNorms + final
        projections trainable. Idempotent: a second call is a no-op, so training
        and load_trainable can each ensure the adapter exists without double-wrap."""
        if getattr(self, "_lora_ready", False):
            return
        t = self.torch
        nn = t.nn
        # NOTE: IRRA's model/petl.py wraps nn.MultiheadAttention.out_proj, which
        # breaks (MHA reads out_proj.weight via its functional path). We inject
        # LoRA only on the transformer MLP linears (c_fc/c_proj) -- those ARE
        # called as modules -- and keep LayerNorms + final projections trainable.
        class LoRALinear(nn.Module):
            def __init__(self, lin, r, alpha):
                super().__init__()
                self.lin = lin
                for p in self.lin.parameters():
                    p.requires_grad_(False)
                self.A = nn.Parameter(t.zeros(r, lin.in_features))
                self.B = nn.Parameter(t.zeros(lin.out_features, r))
                nn.init.normal_(self.A, std=0.02)
                self.scaling = alpha / max(r, 1)
            def forward(self, x):
                return self.lin(x) + (x @ self.A.t() @ self.B.t()) * self.scaling

        class LoRAAttn(nn.Module):
            """Parallel LoRA branch around a whole nn.MultiheadAttention.

            The out_proj linear cannot be swapped for a LoRALinear: MHA reads
            ``out_proj.weight`` inside ``F.multi_head_attention_forward``, not by
            calling the module, so the wrapper is simply never invoked (this is
            the bug in IRRA's petl.py). Wrapping the *whole* attention and adding
            a low-rank branch on its input is equivalent in spirit -- a parallel
            adapter on the attention block -- and it actually runs.
            """

            def __init__(self, mha, r, alpha):
                super().__init__()
                self.mha = mha
                for p in self.mha.parameters():
                    p.requires_grad_(False)
                d = mha.embed_dim
                self.A = nn.Parameter(t.zeros(r, d))
                self.B = nn.Parameter(t.zeros(d, r))
                nn.init.normal_(self.A, std=0.02)
                self.scaling = alpha / max(r, 1)

            def forward(self, q, k, v, **kw):
                out, w = self.mha(q, k, v, **kw)
                return out + (q @ self.A.t() @ self.B.t()) * self.scaling, w

        for p in self.model.parameters():
            p.requires_grad_(False)
        n_lora, n_attn = 0, 0
        for mod in self.model.modules():
            mlp = getattr(mod, "mlp", None)
            if mlp is None:
                continue
            for attr in ("c_fc", "c_proj"):
                lin = getattr(mlp, attr, None)
                if isinstance(lin, nn.Linear):
                    setattr(mlp, attr, LoRALinear(lin, lora_rank, lora_alpha))
                    n_lora += 1
            a = getattr(mod, "attn", None)
            if attn and isinstance(a, nn.MultiheadAttention):
                mod.attn = LoRAAttn(a, lora_rank, lora_alpha)
                n_attn += 1
        for m in self.model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad_(True)
        for nm, p in self.model.named_parameters():
            if nm in ("text_projection", "visual.proj", "logit_scale") or \
               (nm.endswith("proj") and p.dim() <= 2):
                p.requires_grad_(True)
        self.model.to(self.device)                   # move new LoRA params
        self._lora_ready = True
        params = [p for p in self.model.parameters() if p.requires_grad]
        total = sum(p.numel() for p in self.model.parameters())
        tr = sum(p.numel() for p in params)
        print(f"[clip] LoRA on {n_lora} MLP linear(s)"
              + (f" + {n_attn} attention block(s)" if n_attn else "")
              + f" | trainable {tr:,} / {total:,} ({100*tr/max(total,1):.2f}%)")

    # ---- Full fine-tune: unfreeze every backbone parameter ----
    def _enable_full_finetune(self):
        """Unfreeze ALL CLIP parameters for full fine-tuning.

        Unlike LoRA, this trains the entire backbone (~149 M params for ViT-B/16).
        Key implications:
          - Checkpoint contains the *full* model state, not a small adapter.
          - Optimizer (Adam) needs ~1.8 GB extra VRAM for moments + gradients.
          - Learning rate should be 10-50× smaller than LoRA (try 1e-5 – 5e-5).
          - Risk of catastrophic forgetting; use fewer epochs (10-30) and
            gradient clipping.

        Idempotent: calling twice is a no-op.
        """
        if getattr(self, "_full_ft_ready", False):
            return
        for p in self.model.parameters():
            p.requires_grad_(True)
        self._full_ft_ready = True
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[clip] full fine-tune | ALL {total:,} params trainable"
              f" ({total/1e6:.2f} M) | ⚠  use lr ≤ 5e-5 to avoid forgetting")

    # ---- resume: load a previously trained adapter as the starting point ----
    def load_trainable(self, ckpt_path, lora_rank=16, lora_alpha=16,
                       strict=True, full_finetune=None):
        """Copy an adapter checkpoint into the model.

        ``full_finetune`` selects the parameter layout to load INTO.
        ``None`` (default) sniffs it from the checkpoint via
        :func:`is_full_finetune_ckpt`, which is what every caller should use --
        a full fine-tune checkpoint loaded through the LoRA path matches zero
        tensors, and a LoRA checkpoint loaded through the full path matches
        zero tensors, and only one of those two mistakes is obvious from the
        filename.

        ``strict=True`` (the default, and what evaluation must use) raises if
        any LoRA tensor fails to load. ``strict=False`` restores the old lenient
        behaviour, which is correct only when *resuming training* from a
        checkpoint whose parameter set deliberately differs.

        Why the default flipped
        -----------------------
        Lenient loading skips shape-mismatched tensors and carries on. Give it a
        rank-32 checkpoint with ``lora_rank=16`` and every A/B matrix mismatches,
        every one is skipped, and you are left holding **zero-shot CLIP** -- which
        then produces a complete, plausible, entirely wrong row in the results
        table. The failure is invisible precisely where it matters most: four
        different adapters all scoring 9.73 R@1 look like "our method is weak",
        not like "nothing was loaded".

        Use :func:`infer_lora_rank` instead of passing a rank you believe in.
        """
        t = self.torch
        ck = t.load(ckpt_path, map_location=self.device)
        state = ck.get("trainable", ck) if isinstance(ck, dict) else ck
        if full_finetune is None:
            full_finetune = is_full_finetune_ckpt(state)
        if full_finetune:
            # No wrapper modules: the checkpoint's keys ARE the model's keys.
            self._enable_full_finetune()
            print(f"[clip] loading FULL fine-tuned backbone "
                  f"({len(state)} tensor(s)) -- LoRA not injected")
        else:
            self._inject_lora(lora_rank, lora_alpha)
        own = dict(self.model.named_parameters())
        loaded, missing, mismatch = 0, [], []
        with t.no_grad():
            for nm, tensor in state.items():
                p = own.get(nm)
                if p is None:
                    missing.append(nm)
                    continue
                if tuple(p.shape) != tuple(tensor.shape):
                    mismatch.append((nm, tuple(tensor.shape), tuple(p.shape)))
                    continue
                p.copy_(tensor.to(p.device))
                loaded += 1
        skipped = len(missing) + len(mismatch)
        print(f"[clip] resumed {loaded} tensor(s) from {ckpt_path}"
              + (f"  ({skipped} skipped)" if skipped else ""))

        if strict and skipped:
            det = ""
            if missing and not full_finetune and any(
                    not m.endswith((".A", ".B")) and ".lin." not in m
                    for m in missing[:5]):
                det = ("\n  the unknown keys have no '.lin.' infix, which is "
                       "what a FULL fine-tune checkpoint looks like. Pass "
                       "full_finetune=True (eval_protocol: --full-finetune).")
            if mismatch:
                nm, got, want = mismatch[0]
                det = (f"\n  first mismatch: {nm} checkpoint{got} vs model{want}"
                       f"\n  -> the checkpoint's LoRA rank is "
                       f"{infer_lora_rank(state) or '?'}, you asked for "
                       f"{lora_rank}")
            raise RuntimeError(
                f"{skipped} of {len(state)} tensor(s) did not load from "
                f"{ckpt_path} ({len(mismatch)} shape mismatch, {len(missing)} "
                f"unknown key).{det}\n"
                f"Loading the rest would leave the model partly (or entirely) "
                f"zero-shot while still reporting numbers. Pass the right "
                f"--lora-rank/--lora-alpha, or strict=False if you really mean "
                f"to resume a differently-shaped adapter.")
        if strict and loaded == 0:
            raise RuntimeError(f"nothing loaded from {ckpt_path}: the model "
                               f"is still zero-shot CLIP")
        return ck.get("epoch") if isinstance(ck, dict) else None

    # ---- LoRA / full fine-tune (symmetric InfoNCE on the paired set) ----
    def train_lora(self, pairs, epochs=40, lr=5e-4, lora_rank=16, lora_alpha=16,
                   temp=0.07, seed=0, batch=4, amp=True, ckpt_path=None,
                   hard_neg_weight=0.5, margin=0.2, deterministic=False,
                   on_epoch=None, occ_prob=0.0, occ_variants=("torso", "cutout",
                   "foliage"), occ_levels=(0.25, 0.5, 0.7), oil_root=None,
                   loss="infonce", num_instance=4, sdm_temp=0.02,
                   sdm_weight=1.0, lora_attn=False,
                   lr_schedule="cosine", warmup_epochs=5,
                   eval_fn=None, eval_every=0, on_eval=None,
                   select_metric="i2t_R@1",
                   full_finetune=False, grad_clip=1.0,
                   id_loss_weight=0.0, mlm_loss_weight=0.0, cmt_depth=4,
                   train_aug=False, sampler="auto",
                   lr_factor=1.0, bias_lr_factor=1.0, weight_decay=1e-4,
                   weight_decay_bias=None, optimizer="adamw"):
        """``on_epoch(ep, avg_loss)`` -- optional callback fired after every
        epoch (used by train_dataset to stream the loss curve to W&B).

        Learning-rate schedule
        ----------------------
        ``lr_schedule="cosine"`` (default) ramps the LR linearly from 0 over
        ``warmup_epochs`` epochs, then decays it to 0 on a cosine curve -- the
        schedule IRRA / TBPS-CLIP / RDE all use. The previous behaviour (a flat
        ``lr`` for every epoch) is ``lr_schedule="constant"``; it is kept only
        to reproduce older runs. A constant 5e-4 held for 60 epochs never
        settles: the adapter keeps taking full-size steps around the minimum,
        so the *last* epoch is a random draw from that oscillation rather than
        the best point on the curve.

        Best-checkpoint selection
        -------------------------
        Without ``eval_fn`` this method does what it always did: overwrite
        ``ckpt_path`` after every epoch, so the file you are left with is the
        LAST epoch. That is only the best epoch if the run never overfits,
        which on a full split it does.

        Pass ``eval_fn(ep) -> dict`` (metrics) plus ``eval_every=N`` and the
        loop scores the held-out set every N epochs, keeps the parameters that
        maximise ``select_metric``, writes THOSE to ``ckpt_path``, and restores
        them into the live model before returning -- so whatever the caller
        measures next, and whatever the next curriculum stage trains on, is the
        best adapter rather than the final one. The last epoch is still kept
        alongside as ``<stem>_last.pt`` for diagnostics.

        ``on_eval(ep, metrics, is_best)`` fires after each mid-training eval.

        Occlusion augmentation (``occ_prob`` > 0) degrades each image with
        probability ``occ_prob``, drawing a variant and a level uniformly. It is
        applied on the fly rather than precomputed so an image is seen behind a
        different occluder each epoch -- precomputing would let the model
        memorise one occluder per image, which is the failure mode this
        augmentation exists to avoid.

        The re-seed is per (epoch, index), so a run is reproducible without the
        augmentation being *identical* every epoch.
        """
        t = self.torch
        nn = t.nn
        # fix torch RNG so LoRA init (and any torch-side sampling) is repeatable
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)
        if deterministic:
            # exact bit-for-bit reproducibility (slower); pair with --no-amp
            t.backends.cudnn.deterministic = True
            t.backends.cudnn.benchmark = False
        # ---- parameter setup: LoRA adapter OR full backbone ---- #
        if full_finetune:
            self._enable_full_finetune()
        else:
            # Inject LoRA ONCE (idempotent); later calls (e.g. multi-trial
            # training, or a prior load_trainable) keep training the same
            # adapted model instead of wrapping LoRA inside LoRA.
            self._inject_lora(lora_rank, lora_alpha, attn=lora_attn)
        params = [(f"backbone.{nm}", p)
                  for nm, p in self.model.named_parameters() if p.requires_grad]
        # `opt` is built further down, once the identity labels are in hand: the
        # ID classifier's output dimension is the number of identities, and its
        # parameters have to be in the optimizer from step 0.

        imgs = [p["img_path"] for p in pairs]
        txts = [p["prompt"] for p in pairs]
        # Optional identity grouping. When many pairs share ONE caption (NOMAD:
        # ~90 crops of the same actor, all captioned from that actor's single
        # metadata record), two of them in the same batch are catastrophic, not
        # merely noisy: in-batch InfoNCE labels every off-diagonal pair a
        # negative, and the hard-negative margin below then explicitly pushes
        # the most-confusable off-diagonal pair apart -- and that pair IS a true
        # positive. Group-disjoint batching removes those false negatives.
        # Loaders that don't set "group" keep the previous random permutation.
        groups = [p.get("group") for p in pairs]
        groups = groups if any(g is not None for g in groups) else None

        # -- objective ----------------------------------------------------- #
        # Keep the NAME before the batch loop rebinds `loss` to a tensor; the
        # summary dict is built after that loop and would otherwise record
        # "tensor(-4.8e-11, grad_fn=...)" as the objective.
        loss_name = str(loss)
        use_sdm = "sdm" in loss
        use_nce = "infonce" in loss or not use_sdm
        if use_sdm and groups is None:
            raise ValueError(
                "loss='sdm' needs identity labels, but no pair carries a "
                "'group' field. The dataset loader must emit one (json_data "
                "reads person_id/id; parquet sets need the same).")
        if use_sdm and hard_neg_weight:
            # The margin term pushes the hardest off-diagonal pair apart. Under
            # PK sampling that pair is usually two crops of the SAME person --
            # exactly what SDM is trying to pull together. Leaving both on makes
            # them cancel.
            print("[clip] loss=sdm: hard-negative margin disabled "
                  "(it fights SDM's same-identity positives)")
            hard_neg_weight = 0.0
        # -- batch sampler -------------------------------------------------- #
        # "auto" reproduces the historical coupling: SDM gets PK, everything
        # else gets identity-disjoint batches. That coupling is PRINCIPLED --
        # SDM's supervision IS repeated identities in a batch, while for
        # InfoNCE the same repetition is a false negative -- but it means an
        # SDM-vs-InfoNCE comparison changes the objective AND the sampler at
        # once. Pin it explicitly to separate the two.
        #
        # 'random' was added 2026-09-15 for IRRA parity: IRRA's own published
        # CUHK/ICFG/RSTP configs run `sampler: random` (logs/*/configs.yaml),
        # i.e. a plain shuffle with NO identity constraint at all. 'disjoint'
        # is close but not the same -- it *forbids* a repeated identity, random
        # merely makes one rare (batch 64 over 11 003 CUHK identities:
        # C(64,2)/11003 ~ 0.18 colliding pairs per batch). Any run that claims
        # parity with IRRA must use 'random'.
        if sampler not in ("auto", "pk", "disjoint", "random"):
            raise ValueError(f"unknown sampler {sampler!r}; choose "
                             "auto | pk | disjoint | random")
        use_pk = (sampler == "pk") or (sampler == "auto" and use_sdm)
        use_random = (sampler == "random")
        if use_pk and groups is None:
            raise ValueError("sampler='pk' needs identity labels, but no pair "
                             "carries a 'group' field")

        pid_of = None
        num_classes = 0
        # ID loss needs exactly the integer identity labels SDM builds. This
        # block used to sit inside `if use_sdm`, so an InfoNCE + ID run reached
        # the loss with pid_of still None. Build them whenever either asks.
        need_pid = use_sdm or bool(id_loss_weight)
        if need_pid and groups is None:
            raise ValueError(
                f"loss={loss!r} with id_loss_weight={id_loss_weight} needs "
                "identity labels, but no pair carries a 'group' field. The "
                "dataset loader must emit one (json_data reads person_id/id; "
                "parquet sets need the same).")
        if need_pid:
            uniq = {g: i for i, g in enumerate(dict.fromkeys(groups))}
            pid_of = [uniq[g] for g in groups]
            num_classes = len(uniq)
        if use_sdm:
            k_avg = len(groups) / max(len(uniq), 1)
            # Report the batching that will ACTUALLY run. Printing "PK sampling
            # P=.. x K=.." unconditionally (as this line used to) is worse than
            # printing nothing: under --sampler disjoint it states the opposite
            # of what happens, and it sat two lines above the truth in every log.
            batching = (f"PK sampling P={max(2, batch // num_instance)} "
                        f"x K={num_instance}" if use_pk
                        else ("random batches, no identity constraint "
                              "(IRRA parity)" if use_random
                              else "identity-disjoint batches (PK is OFF)"))
            print(f"[clip] loss={loss} | {len(uniq)} identity/identities, "
                  f"{k_avg:.1f} sample(s) each | {batching}")
            if sampler == "disjoint":
                # CORRECTION 2026-09-15. This warning used to claim the
                # objective becomes "algebraically InfoNCE". It does not, and
                # the wrong claim reached a results write-up. SDM is a REVERSE
                # KL -- sum_j p_j (log p_j - log q_j) with p = softmax(sims) --
                # while InfoNCE is the forward cross-entropy -log p_positive.
                # Measured on random features at temp 0.02, batch 32, one-hot
                # labels: cos(grad_SDM, grad_InfoNCE) = 0.32 (untrained) to
                # 0.41 (half-trained). Different loss, different gradient.
                #
                # What IS true, and is the real cost: with a one-hot target the
                # loss has an exact zero at p_positive = 1, so it SATURATES --
                # measured -4.8e-11 at the end of the round-A CUHK run, i.e.
                # no gradient for the tail of training. And SDM's actual
                # selling point (a *distribution* over the identity's other
                # crops) never switches on, because no batch ever holds two
                # crops of one person.
                print("[clip] NOTE: --loss sdm with identity-disjoint batches "
                      "-- no batch holds two crops of one identity, so the "
                      "target is one-hot: SDM's distribution-matching term is "
                      "inactive and the loss saturates at 0 once the batch is "
                      "separated (watch for loss ~1e-11). Use --sampler pk to "
                      "engage it, or --sampler random for IRRA parity.")
            elif k_avg < 2:
                print("[clip] WARNING: fewer than 2 samples per identity on "
                      "average -- SDM's label matrix will be near-diagonal and "
                      "it degenerates to InfoNCE")
        # -- IRRA auxiliary heads (train-time only) ------------------------ #
        aux = None
        mask_id = mask_hi = None
        if id_loss_weight or mlm_loss_weight:
            if id_loss_weight and not num_classes:
                raise ValueError("id_loss_weight > 0 but no identity labels "
                                 "were built -- see the 'group' field above")
            aux = _aux_heads_cls()(
                embed_dim=int(self.model.text_projection.shape[1]),
                num_classes=(num_classes if id_loss_weight else 0),
                mlm=bool(mlm_loss_weight), cmt_depth=cmt_depth,
                vocab_size=len(self.tokenizer.encoder)).to(self.device)
            params = params + [(f"aux.{nm}", p)
                               for nm, p in aux.named_parameters()]
            print(f"[clip] aux heads: id={bool(id_loss_weight)} "
                  f"(num_classes={num_classes}) mlm={bool(mlm_loss_weight)} "
                  f"| {aux.n_params():,} TRAIN-ONLY parameter(s), NOT written "
                  f"to the checkpoint")
        if mlm_loss_weight:
            # <|mask|> is index 49405 and the maskable range is 1..49404 -- the
            # three specials (mask, sot, eot) sit at the end of the vocab.
            mask_id = self.tokenizer.encoder["<|mask|>"]
            mask_hi = len(self.tokenizer.encoder) - 3
        # -- optimizer: per-parameter LR, IRRA solver/build.py:8-30 ---------- #
        # IRRA does NOT train every parameter at one LR. Its published config
        # (logs/CUHK_PEDES/configs.yaml) sets lr_factor 5.0 and gives that
        # multiplier to every randomly-initialised module -- the cross-modal
        # transformer, the ID classifier and the MLM head -- while the
        # pretrained CLIP backbone stays at the base LR. Biases get
        # bias_lr_factor 2.0 and weight decay 0.
        #
        # Until 2026-09-15 this line was a single AdamW group at one LR, so the
        # 44.8M randomly-initialised aux parameters crawled at the backbone's
        # 1e-5. That is the most likely reason rounds B and C bought only
        # +0.59 / +0.47 R@1 from ID+IRR/MLM when IRRA's own ablation credits
        # them +2.82 on top of SDM. The defaults below (lr_factor 1.0,
        # bias_lr_factor 1.0, weight_decay 1e-4, AdamW) reproduce every run
        # made before that date bit-for-bit; parity with IRRA is opt-in via
        # --lr-factor 5 --bias-lr-factor 2 --weight-decay 4e-5 --optimizer adam.
        groups_opt, n_scaled = [], 0
        for nm, p in params:
            plr, pwd = lr, weight_decay
            is_aux = nm.startswith("aux.")          # every aux param is random-init
            if is_aux:
                plr = lr * lr_factor
            if nm.endswith(".bias") or nm.endswith("_bias"):
                # weight_decay_bias=None -> biases keep the ordinary decay,
                # which is what every run before 2026-09-15 did. IRRA parity
                # passes 0.0 (configs.yaml weight_decay_bias).
                if weight_decay_bias is not None:
                    pwd = float(weight_decay_bias)
                if not is_aux:                      # IRRA: classifier/mlm_head
                    plr = lr * bias_lr_factor       # override wins over bias
            if plr != lr:
                n_scaled += 1
            # "lr_scale" is what the epoch loop multiplies the scheduled LR by.
            # Writing cur_lr straight into every group (what the loop used to
            # do) would silently erase lr_factor on the first epoch.
            groups_opt.append({"params": [p], "lr": plr, "weight_decay": pwd,
                               "lr_scale": plr / lr if lr else 1.0})
        if optimizer == "adam":
            # IRRA uses Adam with eps 1e-3 -- not a typo in their repo, and it
            # materially changes early steps at lr 1e-5.
            opt = t.optim.Adam(groups_opt, lr=lr, betas=(0.9, 0.999), eps=1e-3)
        elif optimizer == "adamw":
            opt = t.optim.AdamW(groups_opt, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"unknown optimizer {optimizer!r}; "
                             "choose adamw | adam")
        print(f"[clip] optim: {optimizer} lr={lr:g} wd={weight_decay:g}"
              + (f" wd_bias={float(weight_decay_bias):g}"
                 if weight_decay_bias is not None else "")
              + f" | lr_factor={lr_factor:g} (aux heads) "
                f"bias_lr_factor={bias_lr_factor:g} "
                f"| {n_scaled}/{len(groups_opt)} tensor(s) on a scaled LR")

        n = len(pairs)
        batch = max(2, min(batch, n))                     # need >=2 for negatives
        rng = np.random.default_rng(seed)
        use_amp = bool(amp) and self.device == "cuda"     # AMP only helps on CUDA
        try:
            scaler = t.amp.GradScaler("cuda", enabled=use_amp)
        except Exception:
            scaler = t.cuda.amp.GradScaler(enabled=use_amp)
        dev_type = "cuda" if self.device == "cuda" else "cpu"
        print(f"[clip] train: device={self.device} batch={batch} amp={use_amp} "
              f"epochs={epochs}")

        def trainable_state():
            return {nm: p.detach().cpu().clone()
                    for nm, p in self.model.named_parameters() if p.requires_grad}

        def write_ckpt(path, ep, state, extra=None):
            if not path:
                return
            blob = {"epoch": ep, "trainable": state}
            if extra:
                blob.update(extra)
            t.save(blob, path)

        def save_ckpt(ep):
            write_ckpt(ckpt_path, ep, trainable_state())

        def restore_state(state):
            with t.no_grad():
                by_name = dict(self.model.named_parameters())
                for nm, v in state.items():
                    if nm in by_name:
                        by_name[nm].copy_(v.to(by_name[nm].device))

        # ``_last.pt`` sits next to the selected checkpoint. It is what the old
        # code used to leave in ckpt_path, so an overfit run stays diagnosable:
        # comparing best-vs-last tells you whether 60 epochs was too many.
        last_path = None
        if ckpt_path:
            _cp = Path(str(ckpt_path))
            last_path = str(_cp.with_name(_cp.stem + "_last" + _cp.suffix))

        # bool(): `eval_every and ...` would otherwise leak the int 0 into
        # metrics.json's "selected" field
        select = bool(eval_fn and eval_every and eval_every > 0)
        best = {"score": float("-inf"), "epoch": -1, "state": None, "metrics": None}

        # -- LR schedule ---------------------------------------------------- #
        import math as _math
        sched = str(lr_schedule or "cosine").lower()
        if sched not in ("cosine", "constant"):
            raise ValueError(f"unknown lr_schedule {lr_schedule!r}; "
                             "choose 'cosine' or 'constant'")
        # Warmup longer than the run itself would mean the LR never decays.
        warm = max(0, min(int(warmup_epochs), max(epochs - 1, 0)))

        def lr_at(ep):
            if sched == "constant":
                return lr
            if warm and ep < warm:
                return lr * (ep + 1) / (warm + 1)
            prog = (ep - warm) / max(epochs - warm, 1)
            return lr * 0.5 * (1.0 + _math.cos(_math.pi * min(prog, 1.0)))

        print(f"[clip] train: lr={lr:g} schedule={sched}"
              + (f" warmup={warm}ep" if sched == "cosine" else "")
              + (f" | select best by {select_metric} every {eval_every}ep"
                 if select else " | NO mid-training eval: ckpt = LAST epoch"))

        print(f"[clip] train: sampler={sampler} -> "
              + (f"PK  P={max(2, batch // num_instance)} x K={num_instance}"
                 if use_pk else
                 ("random (no identity constraint)" if use_random
                  else "identity-disjoint"))
              + (f" over {len(set(groups))} identity group(s)"
                 if groups is not None else " (no identity labels)"))
        if use_nce:
            print(f"[clip] train: InfoNCE temperature={temp:g}")

        occ_fn = None
        if occ_prob and occ_prob > 0:
            from .occlude_bench import degrade, VARIANTS, OIL_VARIANTS
            bad = [v for v in occ_variants if v not in VARIANTS]
            if bad:
                raise ValueError(f"unknown occlusion variant(s) {bad}; "
                                 f"choose from {VARIANTS}")
            if any(v in OIL_VARIANTS for v in occ_variants) and not oil_root:
                raise ValueError("occ variant 'ogor' needs oil_root")
            occ_variants, occ_levels = list(occ_variants), list(occ_levels)

            def occ_fn(im, key):                              # noqa: F811
                r = np.random.default_rng(key)
                if r.random() >= occ_prob:
                    return im
                v = occ_variants[int(r.integers(len(occ_variants)))]
                lv = float(occ_levels[int(r.integers(len(occ_levels)))])
                return degrade(im, v, lv, int(r.integers(1 << 31)),
                               oil_root=oil_root)

            print(f"[clip] train: occlusion aug p={occ_prob} "
                  f"variants={occ_variants} levels={occ_levels}")

        if train_aug:
            print("[clip] train: image aug ON (hflip 0.5, pad-10 crop, "
                  "random-erasing 0.5) -- TRAIN PATH ONLY")

        self.model.train()
        for ep in range(epochs):
            cur_lr = lr_at(ep)
            for g in opt.param_groups:
                g["lr"] = cur_lr * g.get("lr_scale", 1.0)
            tot, steps = 0.0, 0
            # MLM top-1 accuracy over masked positions only. RUN_ISATMP.md tells
            # you to watch this climb past 40%; until now nothing printed it, so
            # a silently dead MLM branch would have cost a whole run unnoticed.
            mlm_ok, mlm_n = 0, 0
            # sampler=random -> pass groups=None so _index_batches falls back to
            # a plain shuffle: that IS IRRA's `sampler: random`.
            batches = (_pk_batches(groups, batch, num_instance, rng) if use_pk
                       else _index_batches(None if use_random else groups,
                                           n, batch, rng))
            for bi in batches:
                if len(bi) < 2:
                    continue
                x = t.stack([
                    self._prep_image(
                        imgs[i],
                        aug=(None if occ_fn is None else
                             (lambda im, _i=i: occ_fn(im, (seed, ep, _i)))),
                        # salt 7 keeps this stream independent of the occlusion
                        # stream above, which is keyed on (seed, ep, i)
                        train_rng=(np.random.default_rng((seed, ep, int(i), 7))
                                   if train_aug else None))
                    for i in bi]).to(self.device)
                tok_cpu = t.stack([self._tok(txts[i]) for i in bi])
                tok = tok_cpu.to(self.device)
                mlm_tok = mlm_lbl = None
                if aux is not None and mlm_loss_weight:
                    # per-(seed, epoch, sample) RNG: IRRA masks with the global
                    # `random` module, which would make --deterministic a lie.
                    ml = [_mask_tokens(tok_cpu[j].tolist(), mask_id, mask_hi,
                                       np.random.default_rng((seed, ep,
                                                              int(bi[j]))))
                          for j in range(len(bi))]
                    mlm_tok = t.tensor([m[0] for m in ml], dtype=tok_cpu.dtype,
                                       device=self.device)
                    mlm_lbl = t.tensor([m[1] for m in ml], dtype=tok_cpu.dtype,
                                       device=self.device)
                with t.autocast(device_type=dev_type, enabled=use_amp):
                    # Keep the SEQUENCES: the MLM branch cross-attends text
                    # tokens to image tokens, so pooling here would throw away
                    # the only thing it consumes. Keep the UN-normalised pooled
                    # vectors too -- the ID classifier must see those (IRRA
                    # build.py:113 feeds i_feats raw; L2-normalising first pins
                    # every logit to the unit sphere and flattens the CE).
                    img_seq = self.model.encode_image(x).float()    # [B,1+N,D]
                    txt_seq = self.model.encode_text(tok).float()   # [B,77,D]
                    fi_raw = self._pool_img(img_seq)
                    ft_raw = self._pool_txt(txt_seq, tok)
                    fi = fi_raw / fi_raw.norm(dim=-1, keepdim=True)
                    ft = ft_raw / ft_raw.norm(dim=-1, keepdim=True)
                    b = len(bi)
                    sims = fi @ ft.t()                       # cosine, diag = positives
                    loss = 0.0
                    if use_nce:
                        logits = sims / temp
                        tgt = t.arange(b, device=self.device)
                        loss = (t.nn.functional.cross_entropy(logits, tgt) +
                                t.nn.functional.cross_entropy(logits.t(), tgt)) / 2
                    if use_sdm:
                        pid_b = t.tensor([pid_of[i] for i in bi],
                                         device=self.device)
                        loss = loss + sdm_weight * sdm_loss(
                            fi, ft, pid_b, t, temp=sdm_temp)
                    # hard-negative margin: push the single most-confusable wrong
                    # pair below the true pair by `margin` (both directions)
                    if hard_neg_weight > 0 and b > 1:
                        pos = sims.diag()
                        off = sims - t.eye(b, device=self.device) * 2.0
                        hard_i = off.max(dim=1).values        # hardest wrong prompt / image
                        hard_t = off.max(dim=0).values
                        relu = t.nn.functional.relu
                        hn = (relu(hard_i - pos + margin).mean() +
                              relu(hard_t - pos + margin).mean()) / 2
                        loss = loss + hard_neg_weight * hn
                    if aux is not None:
                        ce = t.nn.functional.cross_entropy
                        if id_loss_weight:
                            pid_batch = t.tensor([pid_of[i] for i in bi],
                                                 device=self.device)
                            # NOT fi/ft: see the comment at the forward pass.
                            loss = loss + id_loss_weight * (
                                ce(aux.classifier(fi_raw), pid_batch) +
                                ce(aux.classifier(ft_raw), pid_batch)) / 2
                        if mlm_loss_weight:
                            # second pass through the SAME text tower + the same
                            # LoRA -- that is what makes this a regulariser on
                            # the adapter rather than a separate branch.
                            mlm_seq = self.model.encode_text(mlm_tok).float()
                            xf = aux.cross_former(mlm_seq, img_seq, img_seq)
                            scores = aux.mlm_head(xf).float().reshape(
                                -1, aux.vocab_size)
                            mlm_flat = mlm_lbl.reshape(-1)
                            loss = loss + mlm_loss_weight * ce(
                                scores, mlm_flat, ignore_index=0)
                            with t.no_grad():
                                keep = mlm_flat != 0      # 0 == "not masked"
                                if bool(keep.any()):
                                    mlm_ok += int((scores[keep].argmax(dim=-1)
                                                   == mlm_flat[keep]).sum())
                                    mlm_n += int(keep.sum())
                opt.zero_grad()
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    t.nn.utils.clip_grad_norm_([p for _, p in params],
                                               grad_clip)
                scaler.step(opt)
                scaler.update()
                tot += float(loss.item()); steps += 1
            avg_loss = tot / max(steps, 1)
            last_ep = (ep == epochs - 1)
            if not select:
                save_ckpt(ep)                     # legacy path: ckpt = last epoch
            if on_epoch is not None:
                on_epoch(ep, avg_loss)

            msg = ""
            if select and ((ep + 1) % eval_every == 0 or last_ep):
                # eval() disables dropout AND makes the encode_* helpers safe to
                # call re-entrantly; train() must come back on before the next
                # epoch or the rest of the run silently trains in eval mode.
                self.model.eval()
                try:
                    m = eval_fn(ep) or {}
                finally:
                    self.model.train()
                score = float(m.get(select_metric, float("-inf")))
                is_best = score > best["score"]
                if is_best:
                    best.update(score=score, epoch=ep, state=trainable_state(),
                                metrics=dict(m))
                    write_ckpt(ckpt_path, ep, best["state"],
                               {"selected_by": select_metric,
                                "select_score": score, "metrics": dict(m)})
                if on_eval is not None:
                    on_eval(ep, m, is_best)
                msg = (f"  {select_metric} {score:.4f}"
                       f"{'  <- best' if is_best else ''}")

            mlm_msg = f"  mlm_acc {mlm_ok / mlm_n:.3f}" if mlm_n else ""
            if ep % 5 == 0 or last_ep or msg:
                print(f"  epoch {ep:3d}  loss {avg_loss:.4f}  "
                      f"lr {cur_lr:.2e}{mlm_msg}{msg}")

        if select:
            write_ckpt(last_path, epochs - 1, trainable_state())
            if best["state"] is not None:
                restore_state(best["state"])
                print(f"[clip] best epoch {best['epoch']} "
                      f"({select_metric}={best['score']:.4f}) restored and saved "
                      f"to {ckpt_path}; last epoch kept at {last_path}")
                if best["epoch"] < epochs - 1:
                    print(f"[clip] NOTE: best was epoch {best['epoch']} of "
                          f"{epochs} -- the run peaked "
                          f"{epochs - 1 - best['epoch']} epoch(s) before it "
                          f"ended; consider fewer epochs.")
            else:
                save_ckpt(epochs - 1)
        self.model.eval()
        return {"best_epoch": best["epoch"], "best_score": best["score"],
                "best_metrics": best["metrics"], "selected": select,
                "lr_schedule": sched, "warmup_epochs": warm,
                "last_ckpt": last_path,
                # Train-time only. The checkpoint holds the adapter alone, so
                # report these in their own column -- never folded into the
                # parameter-efficiency figure.
                "aux_params_train_only": (aux.n_params() if aux is not None
                                          else 0),
                "id_loss_weight": float(id_loss_weight),
                "mlm_loss_weight": float(mlm_loss_weight),
                "cmt_depth": int(cmt_depth),
                "num_classes": int(num_classes),
                "train_aug": bool(train_aug),
                # The objective and the sampler are two knobs, not one. Record
                # both, or a later reader cannot tell which produced the delta.
                "loss": loss_name,
                "sampler": str(sampler),
                # 2026-10-15: runs with --sampler random used to be logged as
                # "disjoint" here although batching was a plain shuffle.
                "sampler_effective": ("pk" if use_pk else
                                      "random" if use_random else "disjoint"),
                "temp": float(temp),
                "sdm_temp": float(sdm_temp)}


def _index_batches(groups, n, batch, rng):
    """Yield index arrays for one epoch.

    ``groups is None`` -> the original behaviour: a single random permutation cut
    into contiguous slices of ``batch``.

    Otherwise every yielded batch holds at most ONE index per group. Indices are
    bucketed by group, buckets are drained round-robin in a reshuffled order each
    sweep, so an epoch still visits every sample about once while guaranteeing no
    two samples in a batch share a caption/identity. O(n) per epoch.
    """
    order = rng.permutation(n)
    if groups is None:
        for s in range(0, n, batch):
            yield order[s:s + batch]
        return

    from collections import defaultdict
    buckets = defaultdict(list)
    for i in order:
        buckets[groups[i]].append(int(i))
    live = list(buckets)
    while True:
        live = [g for g in live if buckets[g]]
        if len(live) < 2:
            return
        rng.shuffle(live)
        for s in range(0, len(live), batch):
            chunk = live[s:s + batch]
            if len(chunk) < 2:
                continue
            yield np.asarray([buckets[g].pop() for g in chunk])


def _pk_batches(groups, batch, k, rng):
    """P identities x K instances per batch -- the sampler SDM needs.

    SDM's supervision *is* the block structure of the label matrix: two images
    of one person in the same batch tell the model they belong together. Feed it
    identity-disjoint batches (``_index_batches``) and the label matrix is the
    identity matrix, at which point the distribution-matching term is inactive:
    the target is one-hot, and the loss has an exact zero at p_positive = 1, so
    it saturates (measured -4.8e-11 at the end of the round-A CUHK run) and
    stops producing gradient. So this sampler does the opposite: draw
    ``batch // k`` identities and ``k`` images of each.

    An earlier version of this docstring said a one-hot target makes SDM
    "algebraically InfoNCE". That is WRONG and the claim escaped into a results
    write-up before it was checked. SDM is a reverse KL,
    ``sum_j p_j (log p_j - log q_j)`` with ``p = softmax(sims)``; InfoNCE is the
    forward cross-entropy ``-log p_positive``. With one-hot ``q`` at temp 0.02,
    batch 32: cos(grad_SDM, grad_InfoNCE) = 0.32 untrained, 0.41 half-trained --
    related, not identical. Note also that IRRA's own published runs use
    ``sampler: random``, whose batches are one-hot almost every time
    (~0.18 colliding pairs per batch of 64 over 11 003 CUHK identities), so PK
    is an improvement over IRRA here, not parity with it.

    Identities with fewer than ``k`` images left in the epoch are topped up by
    resampling their own images rather than being dropped -- on ICFG the tail of
    identities with 2-3 crops would otherwise never appear in a batch that can
    teach anything about them.
    """
    from collections import defaultdict

    by_g = defaultdict(list)
    for i, g in enumerate(groups):
        by_g[g].append(i)
    all_ids = list(by_g)
    pool = {g: [int(i) for i in rng.permutation(by_g[g])] for g in all_ids}
    p = max(2, batch // max(k, 1))

    live = [g for g in all_ids if pool[g]]
    while len(live) >= 2:
        order = [live[i] for i in rng.permutation(len(live))]
        for s in range(0, len(order), p):
            chunk = [g for g in order[s:s + p] if pool[g]]
            if len(chunk) < 2:
                continue
            idx = []
            for g in chunk:
                take = pool[g][:k]
                pool[g] = pool[g][k:]
                if len(take) < k:                    # top up from the identity
                    extra = by_g[g]
                    take = take + [int(extra[j]) for j in
                                   rng.integers(0, len(extra), k - len(take))]
                idx.extend(take)
            if len(idx) >= 2:
                yield np.asarray(idx)
        live = [g for g in live if pool[g]]


def sdm_loss(fi, ft, pid_idx, torch, temp: float = 0.02, eps: float = 1e-8):
    """Similarity Distribution Matching (Jiang & Ye, CVPR 2023), verbatim.

    InfoNCE says "this caption matches this image, every other image in the
    batch is wrong". On ICFG that is false roughly 11 times over: an identity
    has ~11 crops, and a caption written for one of them describes the person in
    all of them. SDM replaces the one-hot target with the *distribution* of
    identity matches -- ``labels / labels.sum(dim=1)`` -- and minimises the KL
    between the predicted similarity distribution and that target.

    The practical consequence shows up in mINP more than in R@1: InfoNCE can rank
    one crop of the right person first and leave the other ten scattered, and
    R@1 will not notice. That is precisely the failure a search-and-rescue
    system cannot afford, and the reason this loss is the fix for a run whose
    mINP is 5x below the baseline while its R@1 is only 2x below.
    """
    labels = (pid_idx[:, None] == pid_idx[None, :]).float()
    labels = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    logits = fi @ ft.t() / temp
    loss = 0.0
    for lg in (logits, logits.t()):
        pred = torch.softmax(lg, dim=1)
        loss = loss + (pred * (torch.log_softmax(lg, dim=1)
                               - torch.log(labels + eps))).sum(dim=1).mean()
    return loss



# --------------------------------------------------------------------------- #
# Train-time image augmentation                                                #
# --------------------------------------------------------------------------- #
def _geom_aug(im, rng, pad=10):
    """RandomHorizontalFlip(0.5) -> Pad(10) -> RandomCrop(back to same size).

    Same three transforms, same order as IRRA ``datasets/build.py:47``. Applied
    AFTER the resize so the crop window is in final-resolution pixels, which is
    also what IRRA does (its Resize comes first in the Compose).

    ``rng`` is a numpy Generator, not torch's global RNG: augmentation has to be
    reproducible from (seed, epoch, index) or ``--deterministic`` means nothing.
    """
    from PIL import Image as _I, ImageOps as _O
    w, h = im.size
    if rng.random() < 0.5:
        im = im.transpose(_I.FLIP_LEFT_RIGHT)
    if pad > 0:
        im = _O.expand(im, border=pad, fill=0)
        x = int(rng.integers(0, 2 * pad + 1))
        y = int(rng.integers(0, 2 * pad + 1))
        im = im.crop((x, y, x + w, y + h))
    return im


def _random_erase(a, rng, p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3),
                  value=0.0, tries=10):
    """RandomErasing on the NORMALISED array ``a`` of shape (H, W, C).

    ``scale=(0.02, 0.4)`` is IRRA's range (torchvision's default stops at 0.33).

    One deliberate deviation: IRRA passes ``value=mean`` -- the RAW channel mean
    -- into a tensor that Normalize has already centred, so it erases with 0.48
    rather than with the mean. Here the fill is 0.0, which IS the mean after
    normalisation. The intent is the same; note it if you diff against IRRA.
    """
    import numpy as np
    if rng.random() >= p:
        return a
    H, W = a.shape[0], a.shape[1]
    area = float(H * W)
    for _ in range(tries):
        target = area * float(rng.uniform(scale[0], scale[1]))
        ar = float(np.exp(rng.uniform(np.log(ratio[0]), np.log(ratio[1]))))
        eh = int(round(np.sqrt(target * ar)))
        ew = int(round(np.sqrt(target / ar)))
        if 0 < eh < H and 0 < ew < W:
            i = int(rng.integers(0, H - eh + 1))
            j = int(rng.integers(0, W - ew + 1))
            a = a.copy()
            a[i:i + eh, j:j + ew, :] = value
            return a
    return a                      # 10 failed draws: leave the image alone


# --------------------------------------------------------------------------- #
# IRRA auxiliary objectives: ID loss + IRR/MLM                                 #
# --------------------------------------------------------------------------- #
_AUX_HEADS_CLS = None


def _mask_tokens(tokens, mask_id, hi, rng):
    """BERT-style 15% masking, ported from IRRA ``datasets/bases.py:170``.

    ``tokens`` is one tokenised caption (list of ints, length 77). Only real BPE
    tokens are maskable: index 0 is padding and the three specials (<|mask|>,
    <|startoftext|>, <|endoftext|>) sit at the end of the vocab, so the window is
    ``0 < token < hi`` with ``hi = len(encoder) - 3``.

    Of the 15% selected: 80% become <|mask|>, 10% become a random token, 10% are
    left alone -- the model must not learn that "unchanged" means "not asked".

    ``rng`` is a numpy Generator seeded per (seed, epoch, sample). IRRA uses the
    global ``random`` module, which would silently break ``--deterministic``.

    Returns ``(mlm_ids, mlm_labels)`` as plain lists. Label 0 means "ignored",
    matching ``cross_entropy(..., ignore_index=0)``.
    """
    toks = [int(v) for v in tokens]
    labels = []
    for i, tok in enumerate(toks):
        if 0 < tok < hi:
            prob = float(rng.random())
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    toks[i] = mask_id
                elif prob < 0.9:
                    toks[i] = int(rng.integers(1, hi))
                labels.append(tok)
                continue
        labels.append(0)
    if not any(labels) and len(toks) > 1:
        # A caption where nothing was selected contributes no gradient at all,
        # and with ignore_index=0 the whole batch element would be dropped.
        labels[1] = toks[1]
        toks[1] = mask_id
    return toks, labels


def _aux_heads_cls():
    """Return the auxiliary-head module class, defining it on first use.

    This module deliberately has no top-level ``import torch`` -- MockEncoder has
    to work on a box without it -- so the class cannot be declared at import
    time. Defining it lazily costs one dictionary lookup per training run.
    """
    global _AUX_HEADS_CLS
    if _AUX_HEADS_CLS is not None:
        return _AUX_HEADS_CLS
    from collections import OrderedDict
    from torch import nn
    from model.clip_model import Transformer, QuickGELU, LayerNorm

    class _IRRAuxHeads(nn.Module):
        """The heads IRRA trains and then discards.

        They live OUTSIDE ``ClipEncoder.model`` on purpose: ``trainable_state()``
        walks ``self.model.named_parameters()``, so nothing here can leak into a
        retrieval checkpoint. Their cost is a training-time cost and belongs in
        its own column -- folding it into the parameter-efficiency figure would
        make the adapter look more expensive than the full fine-tune, which is
        the opposite of true.

        Init follows IRRA ``model/build.py:26-60`` exactly; the std values there
        are not decorative and PyTorch defaults train noticeably worse.
        """

        def __init__(self, embed_dim, num_classes=0, mlm=False, cmt_depth=4,
                     vocab_size=49408):
            super().__init__()
            self.embed_dim = int(embed_dim)
            self.vocab_size = int(vocab_size)
            self.classifier = None
            if num_classes:
                self.classifier = nn.Linear(self.embed_dim, int(num_classes))
                nn.init.normal_(self.classifier.weight.data, std=0.001)
                nn.init.constant_(self.classifier.bias.data, val=0.0)
            self.mlm = bool(mlm)
            if not self.mlm:
                return
            heads = self.embed_dim // 64
            self.cross_attn = nn.MultiheadAttention(self.embed_dim, heads,
                                                    batch_first=True)
            self.cmt = Transformer(width=self.embed_dim, layers=int(cmt_depth),
                                   heads=heads)
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)
            self.mlm_head = nn.Sequential(OrderedDict([
                ("dense", nn.Linear(self.embed_dim, self.embed_dim)),
                ("gelu", QuickGELU()),
                ("ln", LayerNorm(self.embed_dim)),
                ("fc", nn.Linear(self.embed_dim, self.vocab_size))]))
            scale = self.cmt.width ** -0.5
            proj_std = scale * ((2 * self.cmt.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cmt.width) ** -0.5
            for block in self.cmt.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        def cross_former(self, q, k, v):
            """Masked text tokens (q) attend to image tokens (k, v)."""
            x = self.cross_attn(self.ln_pre_t(q), self.ln_pre_i(k),
                                self.ln_pre_i(v), need_weights=False)[0]
            x = x.permute(1, 0, 2)                        # NLD -> LND
            x = self.cmt(x)
            x = x.permute(1, 0, 2)                        # LND -> NLD
            return self.ln_post(x)

        def n_params(self):
            return sum(p.numel() for p in self.parameters())

    _AUX_HEADS_CLS = _IRRAuxHeads
    return _AUX_HEADS_CLS


def get_encoder(kind: str, **kw):
    if kind == "clip":
        return ClipEncoder(**kw)
    if kind == "irra":
        # Lazy: importing it pulls in the IRRA repo, which only the baseline
        # evaluation needs. See sar_person.irra_encoder.
        from .irra_encoder import IrraEncoder
        return IrraEncoder(**kw)
    return MockEncoder(**kw)
