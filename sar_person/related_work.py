"""Published t2i numbers, keyed by benchmark, for side-by-side comparison.

Two rules govern everything in this file, and both exist because breaking either
one produces a table that looks authoritative and is wrong.

**Rule 1 -- compare only within a benchmark.** R@1 is a property of a
(model, gallery) pair, not of a model. 45.06 on AERI-PEDES and 66.78 on
CUHK-PEDES are not 21 points apart in any meaningful sense: different gallery
size, different identity count, different caption source. So the tables here are
indexed by benchmark and :func:`compare` refuses to cross that line. Benchmarks
with no published protocol (nomad, forestperson) return nothing at all rather
than the nearest-looking numbers.

**Rule 2 -- record where each number came from.** Published values drift between
arXiv v1 and camera-ready, and are frequently *re-run* by later papers rather
than copied. IRRA on ICFG appears as 63.46 in the TAG-PR table and as 63.5 in
several surveys; IRRA on Occluded-ICFG is 57.24, a number IRRA's own authors
never reported -- MGCC produced it by retraining IRRA on their dataset. A row
without a source cannot be checked, and an unchecked row is how a wrong number
survives to camera-ready. ``source`` is therefore mandatory.

Before submission, verify every row you actually cite against the PDF. This
table is for positioning during development, not a substitute for reading.
"""
from __future__ import annotations

from typing import Dict, List

#: metric key -> (published units are percent) all values below are in percent.
METRICS = ("R@1", "R@5", "R@10", "mAP", "mINP")

_TAGPR = "TAG-PR (AAAI-26, arXiv:2511.08369) Tables 3/8/9"
_MGCC = "MGCC (AAAI-24) Table 2 -- retrained by MGCC on their Occluded-* sets"
_PROTO = "EVALUATION_PROTOCOL.md §4.2 (cross-check against the source PDF)"

#: benchmark -> list of published rows. ``None`` means the paper did not report
#: that metric -- never 0, and never silently omitted.
PUBLISHED: Dict[str, List[Dict]] = {
    "cuhk": [
        {"method": "CFine",     "venue": "TIP'23",  "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 69.57, "R@5": 85.93, "R@10": 91.15, "mAP": None,  "mINP": None,  "source": _TAGPR},
        {"method": "TBPS-CLIP", "venue": "AAAI'24", "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 72.66, "R@5": 88.14, "R@10": 92.72, "mAP": 64.97, "mINP": None,  "source": _TAGPR},
        {"method": "IRRA",      "venue": "CVPR'23", "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 73.38, "R@5": 89.93, "R@10": 93.71, "mAP": 66.13, "mINP": 50.2,  "source": _TAGPR},
        {"method": "TAG-CLIP",  "venue": "AAAI'26", "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 74.38, "R@5": 88.30, "R@10": 92.59, "mAP": 67.18, "mINP": None,  "source": _TAGPR},
        {"method": "Propot",    "venue": "2024",    "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 74.89, "R@5": 89.90, "R@10": 94.17, "mAP": 67.12, "mINP": None,  "source": _TAGPR},
        {"method": "RDE",       "venue": "CVPR'24", "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 75.94, "R@5": 90.14, "R@10": 94.12, "mAP": 67.56, "mINP": 51.4,  "source": _TAGPR},
        {"method": "RaSa",      "venue": "IJCAI'23", "backbone": "ALBEF (4xA100)", "params": "100%",
         "R@1": 76.51, "R@5": 90.29, "R@10": 94.25, "mAP": 69.38, "mINP": None,  "source": _PROTO},
        {"method": "APTM",      "venue": "MM'23",   "backbone": "Swin+BERT, MALS 1.5M", "params": "100%",
         "R@1": 76.53, "R@5": 90.04, "R@10": 94.15, "mAP": 66.91, "mINP": None,  "source": _PROTO},
    ],
    "icfg": [
        {"method": "CFine",     "venue": "TIP'23",  "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 60.83, "R@5": 76.55, "R@10": 82.42, "mAP": None,  "mINP": None, "source": _TAGPR},
        {"method": "IRRA",      "venue": "CVPR'23", "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 63.46, "R@5": 80.25, "R@10": 85.82, "mAP": 38.06, "mINP": 7.9,  "source": _TAGPR},
        {"method": "TBPS-CLIP", "venue": "AAAI'24", "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 64.52, "R@5": 80.03, "R@10": 85.39, "mAP": 39.54, "mINP": None, "source": _TAGPR},
        {"method": "Propot",    "venue": "2024",    "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 65.12, "R@5": 81.57, "R@10": 86.97, "mAP": 42.93, "mINP": None, "source": _TAGPR},
        {"method": "TAG-CLIP",  "venue": "AAAI'26", "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 66.78, "R@5": 81.36, "R@10": 86.23, "mAP": 43.60, "mINP": None, "source": _TAGPR},
        {"method": "RDE",       "venue": "CVPR'24", "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 67.68, "R@5": 82.47, "R@10": 87.36, "mAP": 40.06, "mINP": None, "source": _TAGPR},
    ],
    "rstp": [
        {"method": "CFine",     "venue": "TIP'23",  "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 50.55, "R@5": 72.50, "R@10": 81.60, "mAP": None,  "mINP": None, "source": _TAGPR},
        {"method": "IRRA",      "venue": "CVPR'23", "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 60.20, "R@5": 81.30, "R@10": 88.20, "mAP": 47.17, "mINP": 25.3, "source": _TAGPR},
        {"method": "Propot",    "venue": "2024",    "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 61.87, "R@5": 83.63, "R@10": 89.70, "mAP": 47.82, "mINP": None, "source": _TAGPR},
        {"method": "TBPS-CLIP", "venue": "AAAI'24", "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 62.10, "R@5": 81.90, "R@10": 87.75, "mAP": 48.00, "mINP": None, "source": _TAGPR},
        {"method": "TAG-CLIP",  "venue": "AAAI'26", "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 63.85, "R@5": 82.50, "R@10": 88.95, "mAP": 50.22, "mINP": None, "source": _TAGPR},
        {"method": "RDE",       "venue": "CVPR'24", "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 65.35, "R@5": 83.95, "R@10": 89.90, "mAP": 50.88, "mINP": 28.1, "source": _TAGPR},
    ],
    # MGCC's own gallery: ICFG test truncated to its first 10 000 records
    # (10 000 images / 512 ids) with one image per identity occluded. Every row
    # here was produced by MGCC, including the IRRA one.
    "icfg_mgcc": [
        {"method": "Dual-Path", "venue": "TOMM'20", "backbone": "RN50-RN50", "params": "100%",
         "R@1": 29.03, "R@5": 53.29, "R@10": 64.50, "mAP": 14.03, "mINP": None, "source": _MGCC},
        {"method": "LGUR",      "venue": "MM'22",   "backbone": "DeiT-BERT", "params": "100%",
         "R@1": 52.91, "R@5": 70.41, "R@10": 77.01, "mAP": 30.46, "mINP": None, "source": _MGCC},
        {"method": "SSAN",      "venue": "arXiv'21", "backbone": "RN50-LSTM", "params": "100%",
         "R@1": 53.01, "R@5": 71.83, "R@10": 78.90, "mAP": 30.00, "mINP": None, "source": _MGCC},
        {"method": "CFine",     "venue": "TIP'23",  "backbone": "CLIP-BERT", "params": "100%",
         "R@1": 55.55, "R@5": 75.14, "R@10": 82.55, "mAP": 30.94, "mINP": None, "source": _MGCC},
        {"method": "IRRA",      "venue": "CVPR'23", "backbone": "CLIP-B/16", "params": "100%",
         "R@1": 57.24, "R@5": 78.07, "R@10": 83.85, "mAP": 30.42, "mINP": None, "source": _MGCC},
        {"method": "MGCC",      "venue": "AAAI'24", "backbone": "CLIP-B/32", "params": "100%",
         "R@1": 59.28, "R@5": 78.32, "R@10": 84.75, "mAP": 33.30, "mINP": None, "source": _MGCC},
    ],
}

#: Benchmarks with no published protocol at all. Listing them explicitly (rather
#: than letting them fall through to "no data") is what stops someone quietly
#: comparing a NOMAD number against a CUHK table.
NO_PUBLISHED = {
    "nomad": "aerial SAR benchmark built in this project; 20 held-out actors, "
             "no published protocol exists. Compare against your own baselines "
             "(CLIP zero-shot, CLIP full-FT, IRRA re-run) only.",
    "forestperson": "no published protocol; 14 test identities. Use for the "
                    "visibility breakdown, not for a SOTA comparison.",
    "cuhk_heldout": "self-made identity split of the CUHK train shards. The "
                    "published CUHK numbers are on a different gallery AND the "
                    "baselines trained on these identities -- see "
                    "EVALUATION_PROTOCOL.md §0c.",
}


def for_benchmark(name: str) -> List[Dict]:
    """Published rows for ``name``; empty list if there is no published protocol."""
    return list(PUBLISHED.get(name, []))


def note_for(name: str) -> str | None:
    return NO_PUBLISHED.get(name)


def rsum(row: Dict) -> float | None:
    ks = [row.get(f"R@{k}") for k in (1, 5, 10)]
    return sum(ks) if all(v is not None for v in ks) else None
