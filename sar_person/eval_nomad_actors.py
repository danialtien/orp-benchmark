"""Actor-level NOMAD check: 100 prototypes x 100 profile prompts, cosine.

The Excel run (``sar_person.run``) scores a trained adapter on 20 missing-person
profiles: a 20x20 image-vs-prompt matrix whose diagonal is the truth. This
module is the same picture on NOMAD -- 100 real actors instead of 20 rows of a
spreadsheet -- so "did this LoRA actually get better at matching a written
description to a person seen from a drone" has an answer with five times the
identities and ~9 000 real aerial crops behind it.

How an actor becomes one row
----------------------------
Every crop of the actor is encoded, the embeddings are averaged, and the mean
is re-normalised: an *actor prototype*. Averaging first and normalising second
is the point -- it is the centroid direction of that identity's crops, so a
single blurred or half-occluded frame moves it a little instead of deciding the
whole row (which is what happens if you pick one crop, as the 20x20 val eval in
``eval_protocol`` has to). Crop counts differ per actor (79..150 at a10), so
``n_crops`` is reported per row: a prototype built from 12 crops is noisier than
one built from 120 and the two are not equally trustworthy.

Two directions, both reported
-----------------------------
``i2t``  a person -> which of the 100 descriptions fits them (the diagonal
         accuracy the Excel heatmap shows).
``t2i``  a description -> which of the 100 people it points at. **This is the
         operational direction**: a search-and-rescue query is a witness
         statement, and the answer must be a person.

The tie trap
------------
Two actors can end up with a byte-identical caption -- the attribute vocabulary
is finite and "a man, in a black top, in dark gray colored bottoms" fits more
than one person. Their columns are then indistinguishable and strict R@1 counts
one of them wrong no matter what the model does. Both numbers are therefore
reported: ``t2i_R@1`` (strict, the diagonal) and ``t2i_R@1_caption`` (a hit if
the top-ranked actor carries the *same caption text*). A large gap between them
means the profiles are not discriminative enough -- a data problem, not a model
problem.

Usage
-----
    # after a training run, LoRA vs its own zero-shot baseline
    python -m sar_person.eval_nomad_actors \
        --ckpt ../Dataset/Test/20260810_120000_clip_16/lora_ckpt.pt \
        --compare-zeroshot --out ../Dataset/Result/nomad_actors/run1

    # the VLM-derived captions instead of NOMAD's metadata ones
    python -m sar_person.eval_nomad_actors --ckpt .../lora_ckpt.pt \
        --profiles ../Dataset/NOMAD/crop/vlm/nomad_vlm_profiles.json \
        --out ../Dataset/Result/nomad_actors/run1_vlm

    # no GPU, no weights, real artifacts
    python -m sar_person.eval_nomad_actors --encoder mock --limit-actors 10 \
        --out /tmp/nomad_actors_smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import metrics_viz as mv

log = logging.getLogger("sar_person.eval_nomad_actors")

PROMPT_KEYS = {"default": "prompt_default", "templated": "prompt_templated",
               "templated_neg": "prompt_templated_neg"}


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #

def load_actor_crops(crop_root: Path, altitudes: List[int] = None,
                     min_visibility: int = 0, max_per_actor: int = 0,
                     split: str = "all") -> Dict[str, List[str]]:
    """``crop/annotations.json`` -> {actor: [absolute crop path, ...]}.

    ``max_per_actor`` samples EVENLY OVER THE FLIGHT (every m-th crop in frame
    order), not at random: consecutive NOMAD frames are near-duplicates, so a
    random 20 of 120 can land inside two short time windows and the prototype
    then describes two poses rather than the actor.
    """
    ann_path = crop_root / "annotations.json"
    if not ann_path.is_file():
        raise SystemExit(f"{ann_path} not found -- run scripts/prepare_nomad.py first")
    records = json.loads(ann_path.read_text(encoding="utf-8"))

    by_actor: Dict[str, List[dict]] = defaultdict(list)
    kept = skipped_alt = skipped_vis = skipped_split = 0
    for r in records:
        if split != "all" and r.get("split") != split:
            skipped_split += 1
            continue
        if altitudes and int(r.get("altitude_m", 0)) not in altitudes:
            skipped_alt += 1
            continue
        if int(r.get("visibility", 100)) < min_visibility:
            skipped_vis += 1
            continue
        by_actor[r["person_id"]].append(r)
        kept += 1
    log.info("[data] %d crop(s) kept (%d wrong altitude, %d below visibility, "
             "%d wrong split) across %d actor(s)", kept, skipped_alt,
             skipped_vis, skipped_split, len(by_actor))

    out: Dict[str, List[str]] = {}
    for actor, recs in sorted(by_actor.items()):
        recs.sort(key=lambda r: str(r.get("frame", r["id"])))
        if max_per_actor and len(recs) > max_per_actor:
            step = len(recs) / max_per_actor
            recs = [recs[int(i * step)] for i in range(max_per_actor)]
        out[actor] = [str(crop_root / "images" / r["file_path"]) for r in recs]
    return out


def load_actor_prompts(crop_root: Path, profiles: str = None,
                       prompt_style: str = "default") -> Dict[str, str]:
    """``nomad_profiles.json`` (or the VLM one) -> {actor: caption}."""
    path = Path(profiles) if profiles else (crop_root / "nomad_profiles.json")
    if not path.is_file():
        raise SystemExit(
            f"{path} not found. Either run scripts/prepare_nomad.py, or pass "
            f"--profiles <crop>/vlm/nomad_vlm_profiles.json.")
    data = json.loads(path.read_text(encoding="utf-8"))
    key = PROMPT_KEYS[prompt_style]
    out = {a: rec[key] for a, rec in data.items() if rec.get(key)}
    log.info("[data] %d caption(s) from %s [%s]", len(out), path.name, key)
    return out


# --------------------------------------------------------------------------- #
# prototypes                                                                  #
# --------------------------------------------------------------------------- #

def actor_prototypes(enc, actors: List[str], crops: Dict[str, List[str]],
                     batch: int = 64) -> np.ndarray:
    """Mean of an actor's crop embeddings, re-normalised. One row per actor.

    All crops go through the encoder in ONE flat call rather than per actor:
    a per-actor loop pays the batch-assembly and kernel-launch overhead 100
    times over and, with uneven crop counts, leaves most batches half empty.
    """
    flat, owner = [], []
    for i, a in enumerate(actors):
        for p in crops[a]:
            flat.append(p)
            owner.append(i)
    if not flat:
        raise SystemExit("no crops to encode")
    log.info("[encode] %d crop(s) -> %d actor prototype(s)", len(flat), len(actors))
    emb = enc.encode_images(flat, batch=batch) if _takes_batch(enc.encode_images) \
        else enc.encode_images(flat)
    emb = np.asarray(emb, dtype=np.float32)

    owner = np.asarray(owner)
    proto = np.zeros((len(actors), emb.shape[1]), dtype=np.float32)
    for i in range(len(actors)):
        proto[i] = emb[owner == i].mean(axis=0)
    return proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)


def _takes_batch(fn) -> bool:
    import inspect
    try:
        return "batch" in inspect.signature(fn).parameters
    except (TypeError, ValueError):                     # pragma: no cover
        return False


def encode_prompts(enc, prompts: List[str], batch: int = 64) -> np.ndarray:
    emb = enc.encode_texts(prompts, batch=batch) if _takes_batch(enc.encode_texts) \
        else enc.encode_texts(prompts)
    emb = np.asarray(emb, dtype=np.float32)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)


# --------------------------------------------------------------------------- #
# scoring                                                                     #
# --------------------------------------------------------------------------- #

def caption_groups(prompts: List[str]) -> np.ndarray:
    """Index of the caption TEXT each actor carries (equal text -> equal id)."""
    seen: Dict[str, int] = {}
    return np.asarray([seen.setdefault(p, len(seen)) for p in prompts])


def score(S: np.ndarray, prompts: List[str], ks=(1, 5, 10)) -> dict:
    """Metrics for a square actor x caption cosine matrix (diagonal = truth).

    ``*_caption`` variants credit a hit when the retrieved actor carries the
    same caption STRING as the query -- see "the tie trap" in the module
    docstring. When every caption is unique they are identical to the strict
    numbers, and the report says so via ``n_unique_captions``.
    """
    n = S.shape[0]
    gt = np.arange(n)
    grp = caption_groups(prompts)

    base = mv.compute_metrics(S, list(gt))          # same schema as run.py

    def recall(M, credit, k):
        order = np.argsort(-M, axis=1)[:, :k]
        return float(np.mean([credit(i, order[i]) for i in range(n)]))

    strict = lambda i, top: i in top                                  # noqa: E731
    same_cap = lambda i, top: bool((grp[top] == grp[i]).any())        # noqa: E731

    St = S.T                                        # rows = captions
    out = dict(base)
    for k in ks:
        if k > n:
            continue
        out[f"i2t_R@{k}"] = recall(S, strict, k)
        out[f"t2i_R@{k}"] = recall(St, strict, k)
        out[f"t2i_R@{k}_caption"] = recall(St, same_cap, k)
    ranks = np.array([int(np.where(np.argsort(-St[i]) == i)[0][0]) + 1
                      for i in range(n)])
    out["t2i_MRR"] = float(np.mean(1.0 / ranks))
    out["t2i_median_rank"] = float(np.median(ranks))
    out["n_actors"] = n
    out["n_unique_captions"] = int(len(set(prompts)))
    out["n_actors_sharing_a_caption"] = int(
        sum(1 for g in grp if (grp == g).sum() > 1))
    return out


def build_actor_matrix(enc, crop_root, profiles: str = None,
                       prompt_style: str = "default", altitudes: List[int] = None,
                       min_visibility: int = 0, max_per_actor: int = 40,
                       split: str = "all", limit_actors: int = 0,
                       batch: int = 64) -> dict:
    """Everything between "an encoder" and "a square cosine matrix".

    Split out of :func:`main` so ``train_dataset`` can call the identical
    computation right after training without shelling out to this CLI: same
    prototypes, same captions, same numbers, one process, one model load.

    Returns ``{"S", "actors", "prompts", "crops"}``.
    """
    crop_root = Path(crop_root)
    crops = load_actor_crops(crop_root, altitudes, min_visibility,
                             max_per_actor, split)
    captions = load_actor_prompts(crop_root, profiles, prompt_style)
    actors = sorted(set(crops) & set(captions))
    missing = sorted(set(crops) - set(captions))
    if missing:
        log.warning("[data] %d actor(s) have crops but no caption (skipped): %s",
                    len(missing), ", ".join(missing[:5]))
    if limit_actors:
        actors = actors[:limit_actors]
    if len(actors) < 2:
        raise SystemExit(f"need >= 2 actors with both crops and a caption, "
                         f"got {len(actors)}")
    prompts = [captions[a] for a in actors]
    log.info("[data] %d actor(s), %d crop(s), %d unique caption(s)", len(actors),
             sum(len(crops[a]) for a in actors), len(set(prompts)))

    P = actor_prototypes(enc, actors, crops, batch)
    T = encode_prompts(enc, prompts, batch)
    return {"S": (P @ T.T).astype(np.float32), "actors": actors,
            "prompts": prompts, "crops": crops}


def save_artifacts(S: np.ndarray, actors: List[str], prompts: List[str],
                   crops: Dict[str, List[str]], out: Path, tag: str = "",
                   S_before: np.ndarray = None, suffix: str = "") -> dict:
    """Heatmaps + CSVs + the metrics dict. ``suffix`` namespaces the filenames
    so this can write into a training run's folder without colliding with the
    run's own ``heatmap_after.png`` (which is a different eval entirely)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gt = list(range(len(actors)))
    mv.save_single(S, gt, out / f"heatmap_after{suffix}.png",
                   f"NOMAD actors x profiles [{tag}]")
    if S_before is not None:
        mv.save_single(S_before, gt, out / f"heatmap_before{suffix}.png",
                       "NOMAD actors x profiles (zero-shot)")
        mv.save_pair(S_before, S, gt, out / f"heatmap_before_after{suffix}.png")
    mv.save_csv(S, actors, out / f"similarity_after{suffix}.csv")
    return {
        "model": tag,
        "n_actors": len(actors),
        "n_crops": sum(len(crops[a]) for a in actors),
        "crops_per_actor": {"min": min(len(crops[a]) for a in actors),
                            "max": max(len(crops[a]) for a in actors)},
        "metrics": score(S, prompts),
        "metrics_zeroshot": score(S_before, prompts) if S_before is not None
        else None,
        "per_actor": per_actor_rows(S, actors, prompts, crops),
    }


def per_actor_rows(S: np.ndarray, actors: List[str], prompts: List[str],
                   crops: Dict[str, List[str]]) -> List[dict]:
    """One diagnostic row per actor: is its own caption retrieved, and if not,
    which actor stole it? The confuser column is what turns a bad number into
    an actionable one (usually two actors in near-identical dark clothing)."""
    n = len(actors)
    St = S.T
    rows = []
    for i in range(n):
        order = np.argsort(-St[i])
        rank = int(np.where(order == i)[0][0]) + 1
        best = int(order[0])
        rows.append({
            "actor": actors[i],
            "n_crops": len(crops[actors[i]]),
            "self_sim": round(float(S[i, i]), 4),
            "t2i_rank": rank,
            "top1_actor": actors[best],
            "top1_sim": round(float(St[i, best]), 4),
            "margin": round(float(S[i, i] - St[i, best]), 4),
            "caption_shared_with": [actors[j] for j in range(n)
                                    if j != i and prompts[j] == prompts[i]],
            "caption": prompts[i],
        })
    return rows


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[2] / "Dataset" / "NOMAD"
    ap.add_argument("--crop-root", default=str(root / "crop"))
    ap.add_argument("--profiles", default=None,
                    help="captions json (default <crop-root>/nomad_profiles.json; "
                         "pass the vlm/ one to score VLM captions)")
    ap.add_argument("--prompt-style", default="default",
                    choices=list(PROMPT_KEYS))
    ap.add_argument("--altitudes", type=int, nargs="*", default=None,
                    metavar="M", help="e.g. --altitudes 10 (default: all "
                                      "altitudes present in the crop set)")
    ap.add_argument("--min-visibility", type=int, default=0)
    ap.add_argument("--split", default="all", choices=["all", "train", "val"],
                    help="'all' (default) is correct when the adapter never saw "
                         "NOMAD. Use 'val' if you trained with --datasets nomad, "
                         "otherwise the score is leakage.")
    ap.add_argument("--max-crops-per-actor", type=int, default=40,
                    help="evenly spaced over the flight; 0 = every crop "
                         "(default 40 -- the prototype is stable well before "
                         "that and encoding 9k crops costs minutes)")
    ap.add_argument("--limit-actors", type=int, default=0)
    # model
    ap.add_argument("--encoder", choices=["clip", "mock"], default="clip")
    ap.add_argument("--pretrain", default="ViT-B/16")
    ap.add_argument("--img-size", nargs=2, type=int, default=[224, 224],
                    metavar=("H", "W"),
                    help="input resolution; MUST match training. Every run "
                         "since round A trained at 384 128 (plain resize). "
                         "Default 224 224 (letterbox) only reproduces reports "
                         "made before this flag existed (2026-10-02), which "
                         "silently scored 384x128 adapters at 224x224")
    ap.add_argument("--ckpt", default=None,
                    help="LoRA adapter or full fine-tune checkpoint to score")
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--compare-zeroshot", action="store_true",
                    help="also score the same backbone with no adapter and save "
                         "a before/after heatmap pair")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)

    crop_root = Path(args.crop_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- encoder ---------------------------------------------------------- #
    from .encoders import get_encoder
    if args.encoder == "mock":
        enc = get_encoder("mock")
        tag = "mock"
    else:
        patch = args.pretrain.rsplit("/", 1)[-1] if "/" in args.pretrain else "16"
        enc = get_encoder("clip", pretrain=args.pretrain, device=args.device,
                          stride=int(patch), img_size=tuple(args.img_size))
        tag = f"clip:{args.pretrain}:zeroshot"

    t0 = time.time()
    kw = dict(profiles=args.profiles, prompt_style=args.prompt_style,
              altitudes=args.altitudes, min_visibility=args.min_visibility,
              max_per_actor=args.max_crops_per_actor, split=args.split,
              limit_actors=args.limit_actors, batch=args.batch)

    # Zero-shot FIRST, on the same encoder object: get_eval_encoder injects the
    # adapter in place, so the baseline has to be measured before that happens.
    # One process also guarantees both matrices come from identical crops,
    # identical prompts and identical preprocessing.
    S_before = None
    if args.compare_zeroshot and args.encoder == "clip":
        log.info("[eval] zero-shot baseline")
        S_before = build_actor_matrix(enc, crop_root, **kw)["S"]

    if args.ckpt and args.encoder == "clip":
        from .eval_protocol import get_eval_encoder
        enc, tag = get_eval_encoder("clip", args.pretrain, args.ckpt,
                                    args.lora_rank, args.lora_alpha, args.device,
                                    img_size=tuple(args.img_size))

    res = build_actor_matrix(enc, crop_root, **kw)
    S, actors, prompts, crops = (res["S"], res["actors"], res["prompts"],
                                 res["crops"])

    report = save_artifacts(S, actors, prompts, crops, out, tag, S_before)
    m_after, m_before, rows = (report["metrics"], report["metrics_zeroshot"],
                               report["per_actor"])
    report.update({
        "crop_root": str(crop_root),
        "profiles": str(args.profiles or (crop_root / "nomad_profiles.json")),
        "config": vars(args),
        "wall_seconds": round(time.time() - t0, 1),
    })
    (out / "metrics.json").write_text(json.dumps(report, indent=2,
                                                 ensure_ascii=False),
                                      encoding="utf-8")
    _write_per_actor_csv(rows, out / "per_actor.csv")

    # ---- report ----------------------------------------------------------- #
    def line(label, m):
        print(f"  {label:<10} t2i R@1={m['t2i_R@1']*100:5.2f}  "
              f"R@5={m['t2i_R@5']*100:5.2f}  MRR={m['t2i_MRR']:.3f}  "
              f"med.rank={m['t2i_median_rank']:.0f}  |  "
              f"i2t R@1={m['i2t_R@1']*100:5.2f}  gap={m['pos_neg_gap']:.4f}")

    print(f"\nNOMAD actor eval  [{tag}]  {len(actors)} actors, "
          f"{report['n_crops']} crops")
    if m_before:
        line("zero-shot", m_before)
    line("model", m_after)
    if m_after["n_unique_captions"] < len(actors):
        print(f"  !! only {m_after['n_unique_captions']}/{len(actors)} captions "
              f"are unique ({m_after['n_actors_sharing_a_caption']} actors share "
              f"one). Strict t2i R@1={m_after['t2i_R@1']*100:.2f} vs "
              f"caption-level {m_after['t2i_R@1_caption']*100:.2f} -- the gap is "
              f"a profile-discriminability limit, not a model error.")
    worst = sorted(rows, key=lambda r: -r["t2i_rank"])[:5]
    print("  hardest actors: " + ", ".join(
        f"{r['actor']}(rank {r['t2i_rank']}, lost to {r['top1_actor']})"
        for r in worst))
    print(f"  -> {out}")


def _write_per_actor_csv(rows: List[dict], path: Path) -> None:
    import csv
    cols = ["actor", "n_crops", "self_sim", "t2i_rank", "top1_actor",
            "top1_sim", "margin", "caption_shared_with", "caption"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["actor"], r["n_crops"], r["self_sim"], r["t2i_rank"],
                        r["top1_actor"], r["top1_sim"], r["margin"],
                        " ".join(r["caption_shared_with"]), r["caption"]])


if __name__ == "__main__":
    main()
