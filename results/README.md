# Results

Per-query evaluation outputs behind every number in the paper. With these files,
`analysis/` recomputes all tables, figures and confidence intervals without any
dataset, checkpoint or GPU.

| file | content |
|---|---|
| `per_query/<model>__<benchmark>.npz` | for one model on one benchmark: `cells` (C,) = `clean`, the 15 ORP cells `<geometry>_<NN>` and, where scored, OOF cells `oof_*`; `inp`, `ap` (C, Q) float32 per-query INP and AP; `hit1`, `hit5`, `hit10` (C, Q) uint8 per-query Rank-K hits. Query order = `manifests/evalsets/<benchmark>.json.gz` |
| `query_identities.npz` | `<bench>_qid`: identity index of each query (the bootstrap resampling unit); `<bench>_k`: number of correct gallery images per query; `<bench>_N`: gallery size |
| `registry.json` | for each per-query file: the checkpoint that was scored, the source `metrics.json` of every cell in the original results tree, and the aggregate metrics (used as a consistency check) |
| `aggregate_only_cells.csv` | cells kept only as aggregates (published IRRA on RSTPReid and ICFG-PEDES, used in Fig. 1); their per-query outputs were not saved |
| `reproduced/` | outputs of `analysis/reproduce_paper.py` (109/109 checks pass), `analysis/seed_consistency.py` and `analysis/ci_stability.py` |

Model names: `irra_published` (IRRA's released CUHK-PEDES checkpoint), `irra_init`
(the same backbone weights converted to this repository's format and scored with
this repository's CLIP encoder: 72.99 clean Rank-1 on CUHK-PEDES against 73.16 for
`irra_published`, which uses IRRA's own model code and preprocessing), `clip_zs` (CLIP ViT-B/16 zero-shot), and one
name per training config in `configs/train/`.

Coverage: D+A seeds 1 and 2 have CUHK-PEDES and RSTPReid grids; their ICFG-PEDES
grids are pending (`docs/STATISTICS.md`, section 5). D+A→NOMAD models have clean
CUHK-PEDES / RSTPReid and NOMAD-val.

`tools/collect_per_query.py` (maintainer tool) rebuilds this folder from the original
results tree.
