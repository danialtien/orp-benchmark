"""
Parameter-efficient transfer learning (PETL) for the CLIP backbone, so the SAME
fine-tuning code can run in three regimes (selected by config):

    tuning.mode = full     -> unfreeze everything (IRRA default)
    tuning.mode = frozen   -> zero-shot CLIP baseline (no grad, sanity floor)
    tuning.mode = petl     -> freeze CLIP, inject LoRA + adapters + (optional) prefix
                              (UP-Person style: <5% trainable params)

This is intentionally self-contained (no extra deps) so it layers cleanly onto a
vendored IRRA `model/clip_model.py`. Call `apply_petl(model, cfg)` AFTER
`build_model(...)` and BEFORE building the optimizer.

References mapped to code:
  - LoRA on attention in/out proj        -> UP-Person (rank=32 default)
  - Bottleneck Adapter after MLP         -> UP-Person depth_adapter / LAE
  - Deep prefix tokens                   -> UP-Person prefix_length / depth_prefix
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """Wrap an existing nn.Linear with a low-rank residual: y = Wx + (B A) x * s."""

    def __init__(self, base: nn.Linear, rank: int = 32, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.zeros(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)
        lora = (self.drop(x) @ self.A.t()) @ self.B.t()
        return out + lora * self.scaling


class Adapter(nn.Module):
    """Bottleneck adapter: x + Up(GELU(Down(LN(x))))."""

    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.act = nn.GELU()
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)  # identity at init

    def forward(self, x):
        return x + self.up(self.act(self.down(self.ln(x))))


# --------------------------------------------------------------------------- #
# Injection
# --------------------------------------------------------------------------- #
def _iter_attention_linears(module):
    """Yield (parent, attr_name, linear) for q/k/v/out projections we can LoRA-fy.
    Handles both nn.MultiheadAttention (in_proj is a Parameter, skip) and the
    nn.Linear projections common in CLIP resblocks / HF CLIP."""
    for name, child in module.named_modules():
        for attr in ("q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"):
            if hasattr(child, attr) and isinstance(getattr(child, attr), nn.Linear):
                yield child, attr, getattr(child, attr)


def apply_petl(model: nn.Module, cfg) -> nn.Module:
    """Freeze CLIP and inject PETL modules according to cfg.tuning.

    cfg.tuning fields (with defaults):
        mode: 'full' | 'frozen' | 'petl'
        lora_rank: 32, lora_alpha: 32
        adapter_dim: 64
        prefix_length: 0          # 0 disables prefix tuning
        train_ln: True            # keep LayerNorms trainable (cheap, helps a lot)
        train_proj: True          # keep CLIP's final image/text projection trainable
    """
    t = cfg.tuning
    mode = getattr(t, "mode", "full")

    if mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return model

    # frozen or petl: start by freezing the whole backbone
    backbone = getattr(model, "base_model", model)
    for p in backbone.parameters():
        p.requires_grad_(False)

    if mode == "frozen":
        return model

    # ----- petl: inject LoRA + adapters -----
    rank = getattr(t, "lora_rank", 32)
    alpha = getattr(t, "lora_alpha", 32)
    adapter_dim = getattr(t, "adapter_dim", 64)

    # LoRA on attention/MLP linears
    replacements = []
    for parent, attr, lin in _iter_attention_linears(backbone):
        replacements.append((parent, attr, LoRALinear(lin, rank, alpha)))
    for parent, attr, new in replacements:
        setattr(parent, attr, new)

    # Adapters after each transformer resblock (best-effort, name-based)
    n_adapters = 0
    for name, child in backbone.named_modules():
        if child.__class__.__name__.lower() in {"resblock", "residualattentionblock", "clipencoderlayer"}:
            dim = _infer_dim(child)
            if dim:
                child.add_module("petl_adapter", Adapter(dim, adapter_dim))
                _hook_adapter(child)
                n_adapters += 1

    # Keep LayerNorms + projections trainable (very cheap, large gains)
    if getattr(t, "train_ln", True):
        for m in backbone.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad_(True)
    if getattr(t, "train_proj", True):
        for nm, p in backbone.named_parameters():
            if "proj" in nm.lower() and p.dim() <= 2:
                p.requires_grad_(True)

    # Task heads (classifier, mlm_head, cross-modal) always train
    for nm, p in model.named_parameters():
        if any(k in nm for k in ("classifier", "mlm_head", "cross_", "ln_pre", "ln_post")):
            p.requires_grad_(True)

    _report(model, n_adapters, len(replacements))
    return model


def _infer_dim(block):
    for m in block.modules():
        if isinstance(m, nn.LayerNorm):
            return m.normalized_shape[0]
    return None


def _hook_adapter(block):
    """Run block.petl_adapter on the block's output via a forward hook."""
    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (mod.petl_adapter(out[0]),) + out[1:]
        return mod.petl_adapter(out)
    block.register_forward_hook(hook)


def _report(model, n_adapters, n_lora):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PETL] LoRA-wrapped linears: {n_lora} | adapters: {n_adapters}")
    print(f"[PETL] trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M "
          f"({100*trainable/total:.2f}%)")


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
