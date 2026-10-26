"""Standard text-to-image person-retrieval evaluation (R@K / mAP / mINP).

This is the *only* module whose numbers may go into the paper. It differs from
``metrics_viz.compute_metrics`` in four ways that all matter for comparability:

===================  ============================  ============================
                     metrics_viz (training signal)  eval_protocol (paper)
===================  ============================  ============================
gallery              subsampled n x n              full test gallery
ground truth         one correct column per row    every image of the same pid
"mAP"                mean(1/rank)  -> this is MRR  true mean average precision
mINP                 absent                        present
direction            i2t reported first            t2i (the TBPS convention)
===================  ============================  ============================

Metric definitions (Ye et al., TPAMI 2021, "Deep Learning for Person
Re-identification"; same as IRRA / RDE eval code):

    R@K   = fraction of queries with >=1 same-pid image in the top K
    AP_i  = mean over the positives of (#positives seen so far / rank)
    INP_i = |P_i| / R_i  where R_i is the rank of the HARDEST positive
            (i.e. how cheap it is to retrieve the *last* matching image)

mINP is the metric this paper leans on: search-and-rescue cares about not
missing the person, and occlusion is precisely what pushes the hardest positive
deep into the ranking.

Usage
-----
    # zero-shot CLIP on the full ICFG test gallery
    python -m sar_person.eval_protocol --benchmark icfg --encoder clip \
        --out ../Dataset/Result/tier1/icfg_clip_zeroshot

    # a trained LoRA adapter (rank/alpha auto-read from the run's metrics.json)
    python -m sar_person.eval_protocol --benchmark icfg --encoder clip \
        --ckpt ../Dataset/Test_SAR_Person/stage1_trio/lora.pt \
        --out ../Dataset/Result/tier1/icfg_lora_r16

    # Tier-3 breakdown by real occlusion level / flight altitude
    python -m sar_person.eval_protocol --benchmark nomad --encoder clip \
        --ckpt .../lora.pt --breakdown visibility altitude_m \
        --out ../Dataset/Result/tier3/nomad_lora

    # the published IRRA baseline, scored by THIS protocol (the ±1% milestone)
    python -m sar_person.eval_protocol --benchmark icfg --encoder irra \
        --irra-config ../IRRA/logs/ICFG-PEDES/iira/configs.yaml \
        --out ../Dataset/Result/tier1/icfg_irra

    # smoke test with no GPU and no weights
    python -m sar_person.eval_protocol --benchmark rstp --encoder mock \
        --out /tmp/smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .eval_sets import EvalSet, build as build_evalset

log = logging.getLogger(__name__)

DEFAULT_KS = (1, 5, 10)


# --------------------------------------------------------------------------- #
# Core metrics                                                                 #
# --------------------------------------------------------------------------- #
def rank_metrics(sim: np.ndarray, q_pids: np.ndarray, g_pids: np.ndarray,
                 ks: Sequence[int] = DEFAULT_KS,
                 return_per_query: bool = False,
                 warn_skipped: bool = True) -> Dict:
    """R@K / mAP / mINP for a [n_query, n_gallery] similarity matrix.

    ``sim`` must be float32/float64 -- computing it in fp16 over a 20k gallery
    changes enough ranks to move R@1 by a few tenths of a point.

    Queries whose pid has no gallery image are skipped (and counted), because
    including them silently caps every metric.
    """
    if sim.dtype not in (np.float32, np.float64):
        raise TypeError(f"similarity must be fp32/fp64, got {sim.dtype}. "
                        "Cast after encoding; fp16 ties change the ranking.")
    q_pids = np.asarray(q_pids).astype(str)
    g_pids = np.asarray(g_pids).astype(str)
    if sim.shape != (len(q_pids), len(g_pids)):
        raise ValueError(f"sim {sim.shape} != ({len(q_pids)}, {len(g_pids)})")

    # stable sort so ties resolve identically across runs / machines
    order = np.argsort(-sim, axis=1, kind="stable")
    matches = g_pids[order] == q_pids[:, None]          # bool [Nq, Ng]

    answerable = matches.any(axis=1)
    n_skipped = int((~answerable).sum())
    if n_skipped and warn_skipped:
        log.warning("[metrics] %d/%d query/queries have no gallery match and "
                    "were excluded", n_skipped, len(q_pids))

    idx = np.flatnonzero(answerable)
    hits = {k: matches[idx, :k].any(axis=1).astype(np.float64) for k in ks}

    aps = np.empty(len(idx))
    inps = np.empty(len(idx))
    pos_ranks = []                                     # 1-based ranks of every positive
    for j, i in enumerate(idx):
        pos = np.flatnonzero(matches[i])               # 0-based ranks
        n_p = len(pos)
        aps[j] = float(np.mean(np.arange(1, n_p + 1) / (pos + 1)))
        inps[j] = n_p / (pos[-1] + 1)
        if return_per_query:
            pos_ranks.append((pos + 1).tolist())

    out = {f"R@{k}": float(hits[k].mean()) for k in ks}
    out["mAP"] = float(aps.mean())
    out["mINP"] = float(inps.mean())
    # Rsum = R@1+R@5+R@10. Not an independent signal -- it is the headline
    # number in MGCC's tables (it is how they claim a win on Occluded-CUHK
    # while losing R@1 to IRRA), so we report it to be comparable, not because
    # it says anything R@K does not. Same units as R@K: multiply by 100 for the
    # ~233-style figure printed in papers.
    rsum_ks = [k for k in (1, 5, 10) if k in ks]
    if len(rsum_ks) == 3:
        out["Rsum"] = float(sum(out[f"R@{k}"] for k in rsum_ks))
    out["n_query"] = int(len(idx))
    out["n_gallery"] = int(len(g_pids))
    out["n_query_skipped"] = n_skipped
    if return_per_query:
        # pos_ranks (2026-10-26, ORP review): 1-based rank of each positive, so
        # single-observation success can be computed offline: positive j (0-based,
        # in rank order) ranks pos_ranks[j] - j among negatives only.
        out["_per_query"] = {"query_index": idx, "ap": aps, "inp": inps,
                             **{f"hit@{k}": hits[k] for k in ks},
                             "pos_ranks": pos_ranks}
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                     seed: int = 0) -> Dict:
    """Query-level paired bootstrap on two per-query metric vectors.

    Cheap (resamples an already-computed vector, no re-encoding) and it is the
    honest way to answer "is +0.4 R@1 real?". Returns the mean difference, a 95%
    CI, and a two-sided p-value from the sign of the resampled differences.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"paired vectors must align: {a.shape} vs {b.shape}")
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    obs = float(a.mean() - b.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"delta": obs, "ci95_low": float(lo), "ci95_high": float(hi),
            "p_value": float(min(p, 1.0)), "n_boot": n_boot}


# --------------------------------------------------------------------------- #
# Encoding (with an on-disk feature cache)                                     #
# --------------------------------------------------------------------------- #
def _fingerprint(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _encode(fn, items, batch):
    try:
        return fn(items, batch=batch)
    except TypeError:                                   # MockEncoder takes no batch
        return fn(items)


def encode_set(enc, es: EvalSet, batch: int = 64,
               cache_dir: str | Path | None = None,
               model_tag: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Encode gallery images and query texts; returns (G [Ng,d], Q [Nq,d]) fp32.

    Text features are cached separately from image features on purpose: the six
    occlusion variants of a benchmark share one query set, so the text encoder
    runs once instead of seven times.
    """
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    def _cached(kind: str, key: str, produce):
        if not cache:
            return produce()
        f = cache / f"{es.name}_{kind}_{key}.npy"
        if f.is_file():
            log.info("[encode] cache hit %s", f.name)
            return np.load(f)
        arr = produce()
        np.save(f, arr)
        return arr

    g_key = _fingerprint(model_tag, len(es.gallery), es.gallery_paths[:3],
                         es.gallery_paths[-3:])
    q_key = _fingerprint(model_tag, len(es.queries), es.query_texts[:3])

    t0 = time.time()
    G = _cached("img", g_key, lambda: np.asarray(
        _encode(enc.encode_images, es.gallery_paths, batch), dtype=np.float32))
    Q = _cached("txt", q_key, lambda: np.asarray(
        _encode(enc.encode_texts, es.query_texts, batch), dtype=np.float32))
    log.info("[encode] gallery %s, queries %s in %.1fs", G.shape, Q.shape,
             time.time() - t0)

    # defensive re-normalisation: every encoder in this repo already L2-normalises,
    # but a caller passing raw features would silently turn cosine into dot product
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    return G.astype(np.float32), Q.astype(np.float32)


def get_eval_encoder(kind: str, pretrain: str = "ViT-B/16",
                     ckpt: str | None = None, lora_rank: int | None = None,
                     lora_alpha: int | None = None, device: str | None = None,
                     irra_config: str | None = None,
                     irra_ckpt: str | None = None,
                     irra_root: str | None = None,
                     mgcc_root: str | None = None,
                     mgcc_ckpt: str | None = None,
                     mgcc_chunk: int = 256,
                     mgcc_opts: Dict | None = None,
                     img_size=None,
                     full_finetune: bool | None = None):
    """Build the encoder and, if ``ckpt`` is given, load the LoRA adapter.

    ``lora_alpha`` is NOT stored inside the checkpoint. If it is not passed we
    read it from the sibling ``metrics.json`` written by ``train_dataset``;
    loading an adapter at the wrong alpha runs it at a different amplitude than
    it was trained at and quietly degrades every number.

    ``kind="irra"`` loads an official IRRA checkpoint instead -- a published
    baseline scored by this exact function, gallery and metric code, which is
    the only way the comparison in the paper is apples-to-apples. See
    :mod:`sar_person.irra_encoder`.
    """
    from .encoders import get_encoder
    if kind == "mock":
        return get_encoder("mock"), "mock"
    if kind == "irra":
        if not irra_config:
            raise SystemExit(
                "--encoder irra needs --irra-config <IRRA run>/configs.yaml "
                "(the file IRRA's own train.py writes next to best.pth). It "
                "carries img_size / stride_size / loss_names, and building the "
                "model from anything else will not load the checkpoint.")
        enc = get_encoder("irra", config=irra_config, ckpt=irra_ckpt,
                          irra_root=irra_root, device=device)
        return enc, enc.tag
    if kind == "mgcc":
        from .mgcc_encoder import get_encoder as _mgcc
        if not (mgcc_root and mgcc_ckpt):
            raise SystemExit("--encoder mgcc needs --mgcc-root and --mgcc-ckpt")
        enc = _mgcc(mgcc_root, mgcc_ckpt, device=device, chunk=mgcc_chunk,
                    opts=mgcc_opts)
        return enc, f"mgcc:{Path(mgcc_ckpt).parent.name}"
    enc = get_encoder("clip", pretrain=pretrain, device=device,
                      img_size=tuple(img_size or (224, 224)))
    if not ckpt:
        return enc, f"clip:{pretrain}:zeroshot"

    ck = Path(ckpt)
    # Adapter or full backbone? Read it off the tensors, for the same reason
    # rank is read off the tensors below: a full fine-tune checkpoint pushed
    # through the LoRA path matches ZERO keys, and lenient loading would then
    # score zero-shot CLIP under the fine-tuned run's name.
    from .encoders import infer_lora_rank, is_full_finetune_ckpt
    if full_finetune is None:
        full_finetune = is_full_finetune_ckpt(str(ck))
    if full_finetune:
        log.info("[eval] %s is a FULL fine-tuned backbone (no LoRA); loading "
                 "all parameters", ck.name)
        enc.load_trainable(str(ck), strict=True, full_finetune=True)
        return enc, f"clip:{pretrain}:fullft:{ck.parent.name}"

    # A LoRA adapter trained ON TOP of a full fine-tune (e.g. --init-ckpt
    # Docc without --full-finetune) only means something on that backbone.
    # Loading it onto OpenAI CLIP would score a model that never existed under
    # the run's name -- so refuse and point at the merge tool.
    _run = _read_run_config(ck) or {}
    _base = _run.get("base_kind") or (_run.get("train_config") or {}).get(
        "base_kind")
    if _base == "full":
        raise SystemExit(
            f"{ck} is a LoRA adapter trained on top of a FULL fine-tuned "
            f"backbone (init_ckpt={_run.get('init_ckpt')}). Merge it first:\n"
            f"  python scripts/export_merged_ckpt.py merge --lora {ck} "
            f"--out {ck.with_name('merged_full.pt')}\n"
            f"and evaluate the merged file (it is a full checkpoint).")

    found = infer_lora_rank(str(ck))
    if found is not None:
        if lora_rank is not None and int(lora_rank) != found:
            log.warning("[eval] --lora-rank %s but %s actually stores rank %d; "
                        "using %d", lora_rank, ck.name, found, found)
        lora_rank = found
    if lora_rank is None or lora_alpha is None:
        cfg = _read_run_config(ck)
        lora_rank = lora_rank if lora_rank is not None else cfg.get("lora_rank")
        lora_alpha = lora_alpha if lora_alpha is not None else cfg.get("lora_alpha")
    if lora_rank is None or lora_alpha is None:
        raise SystemExit(
            f"cannot determine LoRA rank/alpha for {ckpt}.\n"
            f"Looked for metrics.json next to the checkpoint and found "
            f"rank={lora_rank}, alpha={lora_alpha}.\n"
            f"Pass --lora-rank / --lora-alpha explicitly. Guessing is not safe: "
            f"the adapter contributes (alpha/rank) * B@A, so a wrong alpha "
            f"rescales every update.")
    log.info("[eval] loading adapter %s (rank=%s alpha=%s scaling=%.3f)",
             ck.name, lora_rank, lora_alpha, lora_alpha / max(lora_rank, 1))
    # strict: an adapter that fails to load must stop the run, not quietly
    # produce a zero-shot row in the comparison table
    enc.load_trainable(str(ck), lora_rank=int(lora_rank),
                       lora_alpha=int(lora_alpha), strict=True)
    return enc, f"clip:{pretrain}:lora r{lora_rank} a{lora_alpha}:{ck.parent.name}"


def ckpt_slug(ckpt: str | Path, depth: int = 3) -> str:
    """Folder name built from the last ``depth`` path components of a checkpoint.

    ``.../weight/sweep_l14/trio_r16/lora_ckpt.pt`` -> ``sweep_l14__trio_r16__lora_ckpt``

    Every LoRA run in SAR_Pipeline's weight tree is called ``lora_ckpt.pt``; the
    only thing distinguishing ``trio_r16`` from ``trio_r32`` is the directory
    above it. Naming results after the checkpoint keeps a 12-adapter sweep
    readable in ``ls``, which naming them all ``tier1`` does not.
    """
    p = Path(ckpt)
    parts = [*p.parent.parts[-(depth - 1):], p.stem] if depth > 1 else [p.stem]
    safe = ["".join(c if (c.isalnum() or c in "-.") else "_" for c in s)
            for s in parts if s not in ("", "/", "\\")]
    return "__".join(safe) or p.stem


def _read_run_config(ckpt: Path) -> Dict:
    for cand in (ckpt.parent / "metrics.json", ckpt.with_suffix(".json")):
        if cand.is_file():
            try:
                d = json.loads(cand.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict):
                if "lora_rank" in d:
                    return d
                for v in d.values():                    # nested run report
                    if isinstance(v, dict) and "lora_rank" in v:
                        return v
    return {}


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
def evaluate(enc, es: EvalSet, batch: int = 64, ks=DEFAULT_KS,
             cache_dir=None, model_tag: str = "",
             breakdown: Sequence[str] = (), balance_buckets: bool = False,
             balance_seed: int = 0) -> Dict:
    """Full-gallery evaluation, plus optional per-attribute breakdowns.

    A breakdown *restricts the gallery* to one bucket and re-ranks, rather than
    filtering the ranking post-hoc: "how well does the model do when every
    candidate is 30 m up" is a different question from "where do the 30 m images
    land in the global ranking", and the former is what the Tier-3 table claims.
    """
    # Most encoders are dual-tower: one vector per image, one per caption, and
    # the score is their dot product. Some are not -- MGCC aggregates a learned,
    # attention-weighted similarity over patch x word pairs, which no pair of
    # embedding tables can reproduce. An encoder that says so by exposing
    # ``similarity()`` gets to compute S itself; everything downstream (the
    # ranking, the metrics, the breakdowns, the bootstrap) is unchanged, which
    # is the whole point of routing every baseline through this function.
    if hasattr(enc, "similarity"):
        log.info("[eval] %s scores pairs directly (no dual-tower embedding); "
                 "feature cache disabled", model_tag or type(enc).__name__)
        S = np.asarray(enc.similarity(es.gallery_paths, es.query_texts),
                       dtype=np.float32)
        if S.shape != (len(es.queries), len(es.gallery)):
            raise ValueError(
                f"{type(enc).__name__}.similarity returned {S.shape}, "
                f"expected (n_query, n_gallery) = "
                f"({len(es.queries)}, {len(es.gallery)})")
    else:
        G, Q = encode_set(enc, es, batch=batch, cache_dir=cache_dir,
                          model_tag=model_tag)
        S = (Q @ G.T).astype(np.float32)
    res = rank_metrics(S, es.query_pids, es.gallery_pids, ks=ks,
                       return_per_query=True)
    per_q = res.pop("_per_query")

    out = {
        "benchmark": es.name,
        "model": model_tag,
        # The loader's provenance -- split_source, the official counts it
        # differs from, the annotation file it came from. es.stats() carries
        # none of it, so without this line a reconstructed-split run is
        # indistinguishable from an official one once the log scrolls away,
        # and compare_related has nothing to footnote on.
        "meta": dict(es.meta or {}),
        "stats": es.stats(),
        "overall": res,
        "per_query": {k: (v if k == "pos_ranks" else np.asarray(v).tolist())
                      for k, v in per_q.items()},
    }

    for fld in breakdown:
        buckets = _bucketise(es, fld)
        n_bal = None
        if balance_buckets:
            buckets, n_bal = _balance(buckets, seed=balance_seed)
            log.info("[eval] %s: %d bucket(s) balanced to %d gallery image(s) "
                     "each", fld, len(buckets), n_bal)
        sub = {}
        for label, rows in sorted(buckets.items()):
            if len(rows) < 2:
                continue
            gsub = np.asarray(rows)
            # queries whose identity is absent from this bucket are dropped by
            # rank_metrics; that is intended (a VIPeR query cannot be answered
            # by a Market-only gallery) but it means bucket rows have different
            # query counts, so n_query is reported alongside every bucket.
            m = rank_metrics(S[:, gsub].copy(), es.query_pids,
                             es.gallery_pids[gsub], ks=ks, warn_skipped=False)
            m["n_gallery_in_bucket"] = len(rows)
            if n_bal is not None:
                m["balanced_to"] = n_bal
            sub[label] = m
        if sub:
            out.setdefault("breakdown", {})[fld] = sub
    return out


_VIS_BUCKETS = [(0, 30, "<30%"), (30, 50, "30-50%"), (50, 70, "50-70%"),
                (70, 101, ">70%")]


#: below this many distinct values, a numeric field is reported at full
#: resolution instead of being collapsed into bands
_EXACT_MAX_LEVELS = 10


def _bucketise(es: EvalSet, field: str) -> Dict[str, List[int]]:
    """Group gallery row indices by a metadata field.

    ``visibility`` is banded only when it is nearly continuous. NOMAD annotates
    it at six discrete levels (50/60/70/80/90/100); forcing those through the
    four survey bands merges 70/80/90/100 into one ">70%" cell and 50/60 into
    another, i.e. a six-point occlusion curve collapses to two points -- exactly
    the resolution the analysis needs. So: few distinct values -> report them
    verbatim, sorted numerically.
    """
    vals = [g.get(field) for g in es.gallery]
    present = [v for v in vals if v is not None]
    distinct = {str(v) for v in present}
    band = (field == "visibility" and len(distinct) > _EXACT_MAX_LEVELS)

    out: Dict[str, List[int]] = {}
    for i, v in enumerate(vals):
        if v is None:
            continue
        if band:
            iv = int(v)
            label = next((lb for lo, hi, lb in _VIS_BUCKETS if lo <= iv < hi),
                         "unknown")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            # zero-pad so "60%" sorts before "100%" in the report
            label = f"{int(v):03d}%" if field == "visibility" else f"{v:g}"
        else:
            label = str(v)
        out.setdefault(label, []).append(i)
    return out


def _balance(buckets: Dict[str, List[int]], seed: int = 0
             ) -> tuple[Dict[str, List[int]], int]:
    """Subsample every bucket to the size of the smallest one.

    Without this, a visibility curve confounds two variables. NOMAD's val
    buckets hold 285-421 crops, so the 100%-visible bucket is ranked against a
    18% larger gallery than the 70% one -- and a larger gallery lowers R@1 by
    itself. The resulting curve would mix "harder because more occluded" with
    "harder because more distractors", which is the same error that makes MGCC's
    truncated-ICFG numbers unreadable.

    Deterministic given ``seed``, and the drawn subset is reported, so the
    balanced curve is reproducible.
    """
    if not buckets:
        return buckets, 0
    n = min(len(v) for v in buckets.values())
    out = {}
    for label, rows in buckets.items():
        if len(rows) > n:
            rng = random.Random(f"{seed}:{label}")
            rows = sorted(rng.sample(rows, n))
        out[label] = rows
    return out, n


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", help="icfg | rstp | cuhk | nomad")
    ap.add_argument("--manifest", help="path to an EvalSet json (Tier-2 variants)")
    ap.add_argument("--data-root", default=str(
        Path(__file__).resolve().parents[2] / "Dataset"))
    ap.add_argument("--evalset-cache", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--encoder", choices=["clip", "mock", "irra", "mgcc"],
                    default="clip",
                    help="'irra'/'mgcc' score an official published checkpoint "
                         "through this same protocol")
    ap.add_argument("--pretrain", default="ViT-B/16")
    ap.add_argument("--img-size", nargs=2, type=int, default=[224, 224],
                    metavar=("H", "W"),
                    help="MUST match the training run. 384 128 matches "
                         "IRRA/RDE; 224 224 letterboxes (default)")
    ap.add_argument("--ckpt", default=None, help="LoRA adapter to evaluate")
    ap.add_argument("--irra-config", default=None, metavar="YAML",
                    help="IRRA run's configs.yaml (sits next to best.pth)")
    ap.add_argument("--irra-ckpt", default=None, metavar="PTH",
                    help="IRRA weights (default: best.pth beside the config)")
    ap.add_argument("--irra-root", default=None, metavar="DIR",
                    help="IRRA repo root (default: <project>/IRRA)")
    ap.add_argument("--mgcc-root", default=None, metavar="DIR",
                    help="clone of github.com/littlexinyi/MGCC")
    ap.add_argument("--mgcc-ckpt", default=None, metavar="PTH")
    ap.add_argument("--mgcc-chunk", type=int, default=256,
                    help="query/gallery block size for MGCC's pairwise scoring "
                         "-- lower it if you run out of VRAM")
    ap.add_argument("--mgcc-opts", nargs="*", default=[], metavar="K=V",
                    help="override MGCC options, e.g. Rv=0.4 Rt=0.5")
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--full-finetune", dest="full_finetune",
                    action="store_true", default=None,
                    help="force the --ckpt to be read as a FULL fine-tuned "
                         "backbone (train_dataset --full-finetune) instead of "
                         "a LoRA adapter. Normally unnecessary: the kind is "
                         "detected from the checkpoint's tensors. Pass "
                         "--lora-ckpt to force the other way.")
    ap.add_argument("--lora-ckpt", dest="full_finetune",
                    action="store_false",
                    help="force the LoRA path (see --full-finetune)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    ap.add_argument("--breakdown", nargs="*", default=[],
                    help="gallery metadata fields, e.g. visibility altitude_m")
    ap.add_argument("--balance-buckets", action="store_true",
                    help="subsample every breakdown bucket to the smallest "
                         "one's size. Use it for any curve you intend to read "
                         "as 'the effect of X': NOMAD's visibility buckets hold "
                         "285-421 crops, so an unbalanced curve mixes "
                         "occlusion with gallery size")
    ap.add_argument("--balance-seed", type=int, default=0)
    ap.add_argument("--feature-cache", default=None,
                    help="reuse encoded features across variants (recommended)")
    ap.add_argument("--tag", default=None, help="label used in the results table")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-root", default=None, metavar="DIR",
                    help="with --out-from-ckpt: parent for the derived folder "
                         "(default ../Dataset/Result)")
    ap.add_argument("--out-from-ckpt", action="store_true",
                    help="name the output folder after the checkpoint instead "
                         "of a tier, e.g. weight/sweep_l14/trio_r16/lora_ckpt.pt "
                         "-> <out-root>/sweep_l14__trio_r16__lora_ckpt/<benchmark>. "
                         "Sweeping 12 adapters into 12 folders called 'tier1' "
                         "loses the only thing that distinguishes them.")
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the comparison table/chart against prior work")
    ap.add_argument("--cuhk-json", default=None)
    ap.add_argument("--cuhk-images", default=None)
    ap.add_argument("--nomad-prompt-style", default="default")
    ap.add_argument("--nomad-split", default="val",
                    choices=["val", "train", "all"],
                    help="'val' (default) = the 20 held-out actors; 'all' = "
                         "every actor, which takes the R@1 step from 5 points "
                         "to 1 and is what a per-visibility breakdown needs. "
                         "'all' reports MEMORISATION for a NOMAD-trained "
                         "adapter -- only use it on models trained elsewhere.")
    ap.add_argument("--nomad-crop", default="crop", metavar="DIR",
                    help="prepared crop directory under Dataset/NOMAD/ "
                         "(default 'crop'). The default build used "
                         "--min-visibility 50, so it has NO crops below 50%% "
                         "visibility and cannot show real occlusion past that. "
                         "Use 'crop_all' (prepare_nomad --min-visibility 10).")
    ap.add_argument("--nomad-min-vis", type=int, default=0, metavar="PCT",
                    help="drop gallery crops below this visibility. Can only "
                         "remove what the prepared set has, never recover what "
                         "prepare_nomad already filtered out.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if bool(args.benchmark) == bool(args.manifest):
        raise SystemExit("pass exactly one of --benchmark / --manifest")
    if not args.out and not args.out_from_ckpt:
        raise SystemExit("pass --out <dir>, or --out-from-ckpt to derive it "
                         "from --ckpt. --out-root alone is not enough: it only "
                         "names the PARENT for --out-from-ckpt.")
    if args.out_from_ckpt and not args.ckpt:
        raise SystemExit("--out-from-ckpt needs --ckpt (there is no checkpoint "
                         "path to name a zero-shot run after)")
    # Validated here, not where the encoder is built: building the ICFG eval set
    # takes minutes, and dying on a missing flag afterwards wastes all of it.
    if args.encoder == "irra" and not args.irra_config:
        raise SystemExit(
            "--encoder irra needs --irra-config <IRRA run>/configs.yaml "
            "(written next to best.pth by IRRA's train.py)")
    if args.irra_config and args.encoder != "irra":
        log.warning("[eval] --irra-config is ignored with --encoder %s",
                    args.encoder)
    if args.encoder == "mgcc" and not (args.mgcc_root and args.mgcc_ckpt):
        raise SystemExit("--encoder mgcc needs --mgcc-root <clone of "
                         "github.com/littlexinyi/MGCC> and --mgcc-ckpt")
    mgcc_opts = {}
    for kv in args.mgcc_opts:
        if "=" not in kv:
            raise SystemExit(f"--mgcc-opts expects K=V, got {kv!r}")
        k, v = kv.split("=", 1)
        try:                                    # keep ints/floats/bools typed
            mgcc_opts[k] = json.loads(v)
        except json.JSONDecodeError:
            mgcc_opts[k] = v

    if args.manifest:
        es = EvalSet.load(args.manifest)
    else:
        es = build_evalset(args.benchmark, args.data_root,
                           cache_dir=args.evalset_cache, rebuild=args.rebuild,
                           cuhk_json=args.cuhk_json,
                           cuhk_images=args.cuhk_images,
                           nomad_prompt_style=args.nomad_prompt_style,
                           nomad_split=args.nomad_split,
                           nomad_crop=args.nomad_crop,
                           nomad_min_vis=args.nomad_min_vis)

    enc, tag = get_eval_encoder(args.encoder, args.pretrain, args.ckpt,
                                args.lora_rank, args.lora_alpha, args.device,
                                irra_config=args.irra_config,
                                irra_ckpt=args.irra_ckpt,
                                irra_root=args.irra_root,
                                mgcc_root=args.mgcc_root,
                                mgcc_ckpt=args.mgcc_ckpt,
                                mgcc_chunk=args.mgcc_chunk,
                                mgcc_opts=mgcc_opts,
                                img_size=tuple(args.img_size),
                                full_finetune=args.full_finetune)
    # resolution belongs in the tag: a 224x224 and a 384x128 run are different
    # models, and they must not share a feature-cache entry
    if args.encoder == "clip" and tuple(args.img_size) != (224, 224):
        tag = f"{tag}@{args.img_size[0]}x{args.img_size[1]}"
    tag = args.tag or tag

    t0 = time.time()
    res = evaluate(enc, es, batch=args.batch, ks=tuple(args.ks),
                   cache_dir=args.feature_cache, model_tag=tag,
                   breakdown=args.breakdown,
                   balance_buckets=args.balance_buckets,
                   balance_seed=args.balance_seed)
    res["wall_seconds"] = round(time.time() - t0, 1)
    res["argv"] = vars(args)

    if args.out_from_ckpt:
        root = Path(args.out_root or (Path(__file__).resolve().parents[2]
                                      / "Dataset" / "Result"))
        out = root / ckpt_slug(args.ckpt) / es.name
        log.info("[eval] --out-from-ckpt -> %s", out)
    else:
        out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    o = res["overall"]
    print(f"\n{es.name}  [{tag}]  gallery={o['n_gallery']}  queries={o['n_query']}")
    print("  " + "  ".join(f"{k}={o[k]*100:.2f}" for k in
                           [f"R@{k}" for k in args.ks] + ["mAP", "mINP"]
                           + (["Rsum"] if "Rsum" in o else [])))
    for fld, buckets in (res.get("breakdown") or {}).items():
        print(f"  -- by {fld}")
        for label, m in buckets.items():
            print(f"     {label:>10}  |G|={m['n_gallery_in_bucket']:<6} "
                  f"|Q|={m['n_query']:<6} R@1={m['R@1']*100:5.2f}  "
                  f"mAP={m['mAP']*100:5.2f}  mINP={m['mINP']*100:5.2f}")
    print(f"  -> {out/'metrics.json'}")

    if not args.no_compare:
        # Written next to metrics.json so a result folder is self-contained:
        # the numbers, and what they mean relative to prior work, in one place.
        from .compare_related import build_table, load_runs, write_all
        rows, note = build_table(load_runs([out / "metrics.json"]), es.name)
        info = write_all(rows, es.name, note, out)
        print(f"  -> {info['markdown']}"
              + (f"\n  -> {info['chart']}" if info.get("chart") else ""))
        if note:
            print(f"  note: no published protocol for {es.name} -- the table "
                  f"holds your run(s) only")


if __name__ == "__main__":
    main()
