"""Similarity matrix, retrieval metrics, and Trail-style heatmaps / table."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def similarity(img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
    """rows = images, cols = prompts. Both assumed L2-normalised."""
    return img_emb @ txt_emb.T


def compute_metrics(S: np.ndarray, gt_col: list[int]) -> dict:
    """gt_col[i] = column (prompt index) that truly matches image row i.

    TRAINING SIGNAL ONLY -- do not put these numbers in a paper.

    This assumes exactly ONE correct column per row, so on any dataset where a
    person has several images (CUHK/ICFG/RSTP/NOMAD) the "correct" answers other
    than the paired one are scored as errors. Two consequences:

    * ``i2t_mAP`` is really mean reciprocal rank (mean of 1/rank); it is exposed
      as ``i2t_MRR`` too, and the ``i2t_mAP`` spelling is kept only so older
      metrics.json files keep parsing. Never report it as mAP.
    * mINP is undefined here (it needs the hardest of several positives).

    For paper-grade numbers on the full test gallery with identity-level ground
    truth, use ``sar_person.eval_protocol.rank_metrics``.
    """
    n = S.shape[0]
    gt = np.asarray(gt_col)
    def recall_at_k(M, gt, k):
        order = np.argsort(-M, axis=1)
        return float(np.mean([gt[i] in order[i, :k] for i in range(len(gt))]))
    def mean_ap(M, gt):
        aps = []
        for i in range(len(gt)):
            order = np.argsort(-M[i])
            rank = int(np.where(order == gt[i])[0][0]) + 1
            aps.append(1.0 / rank)
        return float(np.mean(aps))
    pos = np.array([S[i, gt[i]] for i in range(n)])
    off = S.copy().astype(float)
    for i in range(n):
        off[i, gt[i]] = np.nan
    neg = np.nanmean(off, axis=1)
    top1 = float(np.mean([np.argmax(S[i]) == gt[i] for i in range(n)]))
    # text->image direction
    St = S.T
    gtt = np.argsort(gt)  # inverse mapping: for prompt col, the image row
    inv = np.zeros(n, int)
    for i in range(n):
        inv[gt[i]] = i
    return {
        "n": n,
        "i2t_R@1": recall_at_k(S, gt, 1),
        "i2t_R@5": recall_at_k(S, gt, min(5, n)),
        "i2t_MRR": mean_ap(S, gt),
        "i2t_mAP": mean_ap(S, gt),      # misnomer, kept for old metrics.json

        "t2i_R@1": recall_at_k(St, inv, 1),
        "top1_diagonal_acc": top1,
        "mean_positive_sim": float(pos.mean()),
        "mean_negative_sim": float(np.nanmean(neg)),
        "pos_neg_gap": float(pos.mean() - np.nanmean(neg)),
    }


# ---- Trail-style rendering ---------------------------------------------------
_CMAP = "Blues"

def _grid(ax, S, gt_col, title, annotate=False, ann_fontsize=9):
    n = S.shape[0]
    vmin, vmax = float(S.min()), float(S.max())
    ax.imshow(S, cmap=_CMAP, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"Pr{j+1}" for j in range(n)], fontsize=7, rotation=90)
    ax.set_yticklabels([f"P{i+1}" for i in range(n)], fontsize=7)
    ax.set_xlabel("Prompts"); ax.set_ylabel("Images")
    # gridlines
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    # highlight ground-truth cell (correct match) regardless of shuffle
    for i, j in enumerate(gt_col):
        ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False,
                               edgecolor="#d1495b", linewidth=1.8))
    if annotate:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center",
                        fontsize=ann_fontsize, color="white", fontweight="bold")


def save_pair(S_before, S_after, gt_col, path):
    fig, axs = plt.subplots(1, 2, figsize=(max(8, S_before.shape[0]*0.9), max(5, S_before.shape[0]*0.45)))
    _grid(axs[0], S_before, gt_col, "Before (zero-shot)", annotate=True, ann_fontsize=6)
    _grid(axs[1], S_after, gt_col, "After (LoRA fine-tune)", annotate=True, ann_fontsize=6)
    fig.suptitle("CLIP prompt–image similarity (red = correct match)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(path, dpi=150); plt.close(fig)


def save_single(S, gt_col, path, title):
    fig, ax = plt.subplots(figsize=(max(6, S.shape[0]*0.55), max(6, S.shape[0]*0.55)))
    _grid(ax, S, gt_col, title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_table(S, gt_col, path, title="Similarity table (after)"):
    fig, ax = plt.subplots(figsize=(max(7, S.shape[0]*0.7), max(7, S.shape[0]*0.7)))
    _grid(ax, S, gt_col, title, annotate=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_selfsim_bars(S_before, S_after, gt_col, labels, path,
                      title="Per-person self-similarity: CLIP zero-shot vs CLIP + LoRA"):
    """Grouped bar chart of self-similarity (image <-> own prompt) per person,
    before (zero-shot) vs after (LoRA), with dashed mean lines."""
    n = S_before.shape[0]
    before = np.array([S_before[i, gt_col[i]] for i in range(n)], dtype=float)
    after = np.array([S_after[i, gt_col[i]] for i in range(n)], dtype=float)
    c_before, c_after = "#d0592c", "#3aa06e"
    x = np.arange(n)
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(10, n * 0.65), 5.5))
    ax.bar(x - w/2, before, w, color=c_before, label="CLIP zero-shot (before)")
    ax.bar(x + w/2, after, w, color=c_after, label="CLIP + LoRA (after)")
    mb, ma = float(before.mean()), float(after.mean())
    ax.axhline(mb, color=c_before, linestyle="--", linewidth=1.4)
    ax.axhline(ma, color=c_after, linestyle="--", linewidth=1.4)
    ax.text(n - 0.4, mb, f"mean {mb:.2f}", color=c_before, fontsize=9,
            va="bottom", ha="right")
    ax.text(n - 0.4, ma, f"mean {ma:.2f}", color=c_after, fontsize=9,
            va="bottom", ha="right")
    # compact tick labels: "Person 7" -> "P7"; shared meaning goes in the x-label
    import re
    ticks = ([re.sub(r"(?i)^person\s*", "P", str(l)) for l in labels]
             if labels else [f"P{i+1}" for i in range(n)])
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, fontsize=8, rotation=0 if n <= 25 else 90)
    ax.set_xlabel("Person")
    ax.set_ylabel("Self-similarity (image ↔ own prompt)")
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, max(1.0, float(max(before.max(), after.max())) * 1.1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_csv(S, labels, path):
    import csv
    n = S.shape[0]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"Pr{j+1}" for j in range(n)])
        for i in range(n):
            w.writerow([labels[i]] + [f"{S[i,j]:.4f}" for j in range(n)])


def save_per_image_json(pairs, S_before, S_after, prompt_order, gt_col, path, meta):
    """Per-image record: filename, path, default prompt, the prompt the LoRA model
    best-matches (top-1), the zero-shot match for comparison, and timestamps."""
    import os, json, datetime
    n = len(pairs)
    prompts = [pairs[k]["prompt"] for k in prompt_order]   # columns (possibly shuffled)

    def mtime(p):
        try:
            return datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")
        except Exception:
            return None

    recs = []
    for i in range(n):
        jb = int(np.argmax(S_before[i]))
        ja = int(np.argmax(S_after[i]))
        recs.append({
            "index": i + 1,
            "label": pairs[i]["label"],
            "filename": os.path.basename(pairs[i]["img_path"]),
            "path": pairs[i]["img_path"],
            "image_mtime": mtime(pairs[i]["img_path"]),
            "prompt_default": pairs[i]["prompt"],
            "self_similarity_before": round(float(S_before[i, gt_col[i]]), 4),
            "self_similarity_after": round(float(S_after[i, gt_col[i]]), 4),
            "match_before_finetune": {
                "prompt": prompts[jb],
                "matched_label": pairs[prompt_order[jb]]["label"],
                "similarity": round(float(S_before[i, jb]), 4),
                "is_correct": bool(jb == gt_col[i]),
            },
            # prompt the LoRA fine-tuned model associates most with this image:
            "prompt_lora_finetune": {
                "prompt": prompts[ja],
                "matched_label": pairs[prompt_order[ja]]["label"],
                "similarity": round(float(S_after[i, ja]), 4),
                "is_correct": bool(ja == gt_col[i]),
            },
        })
    out = {**meta, "images": recs}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
