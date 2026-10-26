# Metrics, confidence intervals and seed consistency

Everything on this page is recomputed by three scripts from the released per-query
results (`results/per_query/*.npz`). They need only NumPy: no dataset, checkpoint or GPU.

```bash
python analysis/reproduce_paper.py     # every number printed in the paper (109 checks)
python analysis/seed_consistency.py    # main findings, seed by seed
python analysis/ci_stability.py        # Monte-Carlo stability of the CIs; identity vs. query resampling
```

Their outputs are also shipped in `results/reproduced/`.

## 1. Metrics

For a text query *q* with |G_q| correct gallery images, let R_q^hard be the rank of the
*last* correct image (the hardest one to find).

| quantity | definition |
|---|---|
| INP_q | \|G_q\| / R_q^hard; mINP is its mean over queries |
| Rank-1 | 1 if the top-ranked image shows the queried person |
| retention | Ret_M(g, c) = M(g, c) / M(clean), for metric M, geometry g and coverage c |
| AURC | 100 × mean over the 3 geometries of the trapezoid area of Ret_M over c = 0.1, ..., 0.5, divided by 0.4. The clean point is only the denominator, not a node of the trapezoid |
| Div | Ret_Rank-1(g, c) / Ret_mINP(g, c). Div > 1 means Rank-1 retention overstates how well the hardest image survives |
| chance mINP | expected mINP of a random ranking, computed exactly from each query's \|G_q\| and the gallery size (`chance_minp` in `analysis/reproduce_paper.py`) |

All per-query values are produced by `sar_person/eval_protocol.py` (`rank_metrics`). The
gallery and query sets are those in `manifests/evalsets/`.

## 2. How the confidence intervals are computed

Every CI in the paper is a **95% paired, identity-level (cluster) bootstrap percentile
interval**. The code is `paired_bootstrap` in `analysis/orp.py`.

1. **Unit of resampling: the test identity.** Queries of one person are not
   independent: they share the same gallery images, so a hard person makes all of
   that person's queries hard. Resampling queries one by one would treat correlated
   queries as independent evidence and make the interval too narrow. Each replicate
   therefore draws the test identities with replacement (CUHK-PEDES 1,000,
   ICFG-PEDES 1,000, RSTPReid 200), and every query inherits the weight of its identity.
2. **Paired.** The same resampled identities are applied to both models being
   compared, so the interval is for the *difference* between the models on the same
   test population. Most of the identity-to-identity variation cancels out.
3. **Everything is recomputed inside each replicate.** The clean value, all 15 cell
   values and the AURC are recomputed from the reweighted queries in every replicate.
   Retention is thus a ratio of two weighted means, not a ratio of fixed numbers.
4. **B = 2,000 replicates.** The interval is the 2.5th to 97.5th percentile of the 2,000
   replicate differences. The reported point estimate is the full-sample difference,
   not the bootstrap mean.
5. **NOMAD-val** has 20 held-out actors and one query per actor, so its intervals use a
   plain percentile bootstrap of the mean per-actor difference over the 20 actors
   (B = 10,000; `mean_bootstrap` in `analysis/orp.py`).

**What the intervals cover.** They describe uncertainty from the finite test set
(which identities happen to be in it), for one trained model. They do **not** include
training randomness. Training randomness is measured separately, by training D+A with
three seeds (Section 4). The two sources are reported separately rather than combined.

**Monte-Carlo stability.** `analysis/ci_stability.py` redraws each interval with five
RNG seeds at B = 2,000 and once at B = 10,000. Endpoints move by at most 0.16
percentage points; no conclusion depends on a difference that small:

| comparison | identity bootstrap (5 RNG seeds, B = 2,000) | B = 10,000 | query-level bootstrap (not used) |
|---|---|---|---|
| CUHK-PEDES, mINP-AURC, D+A − D | +10.05, low 9.34–9.40, high 10.69–10.72 | [9.38, 10.70] | [9.56, 10.56] |
| CUHK-PEDES, clean Rank-1, D+A − D+10 | +0.18, low −0.73 to −0.66, high 1.03–1.07 | [−0.69, 1.05] | [−0.63, 0.99] |
| CUHK-PEDES, mINP-AURC, IRRA+A − IRRA | +15.08, low 14.26–14.33, high 15.81–15.86 | [14.31, 15.85] | [14.54, 15.56] |
| RSTPReid, mINP-AURC, D+A − D | +10.42, low 8.34–8.45, high 12.50–12.66 | [8.37, 12.54] | [9.13, 11.67] |
| ICFG-PEDES, mINP-AURC, D+A − D | +9.34, low 8.16–8.26, high 10.51–10.56 | [8.21, 10.55] | [8.71, 9.98] |

The last column shows that resampling queries instead of identities gives narrower
intervals in every comparison: 5% narrower for clean Rank-1, 24–46% for mINP-AURC. That
is the over-confidence the identity bootstrap avoids.

The paper's intervals were drawn by the original analysis script with one NumPy
generator shared by all comparisons. The release gives each comparison a fresh
`default_rng(seed)`. The method is the same, but the random stream differs, so a
recomputed endpoint can differ from the printed one by a few hundredths.
`reproduce_paper.py` therefore checks point estimates exactly and CI endpoints within
0.15 points.

## 3. Which runs have seeds

| model | seeds | scored on |
|---|---|---|
| D+A (CUHK-PEDES, occlusion-aware) | 0, 1, 2 | CUHK-PEDES and RSTPReid: clean + 15 cells for each seed. ICFG-PEDES: seed 0 (seeds 1–2 pending, see below) |
| D+A → NOMAD (aerial Stage 2) | 0, 1, 2 | NOMAD-val, CUHK-PEDES clean, RSTPReid clean |
| D, D+10, D+A′, IRRA+A | 0 | single runs, used as fixed references |
| published IRRA, CLIP zero-shot | – | no training involved |

The training seed controls the occlusion draws (re-drawn per seed, epoch and image),
the data order, dropout, and which 300 training identities are held out for
checkpoint selection. The evaluation itself is deterministic: the occluded galleries
are fixed (`manifests/occluded/`) and identical for every model and seed.

## 4. Do the main findings hold for every seed?

Output of `analysis/seed_consistency.py` (values in %, CIs are 95% paired identity bootstraps):

| finding (paper section) | CUHK-PEDES | RSTPReid (unseen) |
|---|---|---|
| F1 Rank-1 retention > mINP retention (Div > 1) in all 15 cells (3.2) | holds for all 3 seeds (min Div 1.014 / 1.020 / 1.023) | holds for all 3 seeds (min Div 1.045 / 1.038 / 1.039) |
| F2 D+A raises mINP-AURC over D; CI excludes 0 (3.3) | holds for all 3 seeds: +10.05 [9.37, 10.69]; +10.09 [9.45, 10.70]; +11.04 [10.35, 11.68] | holds for all 3 seeds: +10.42 [8.37, 12.66]; +9.15 [7.05, 11.18]; +9.08 [6.62, 11.46] |
| F3 The gain is not extra training: D+A > D+10 in mINP-AURC, CI excludes 0 (3.3) | holds for all 3 seeds: +10.73; +10.77; +11.72 (all CIs > 10.0) | holds for all 3 seeds: +9.74; +8.48; +8.41 (all CIs > 5.6) |
| F3 D+A > D+10 in absolute mINP in all 15 cells (3.3) | holds for all 3 seeds (15/15 each) | holds for all 3 seeds (15/15 each) |
| F4 No clean Rank-1 cost relative to D+10 (CI upper bound > 0) (3.3) | holds for all 3 seeds: +0.18 [−0.66, 1.04]; +0.44 [−0.44, 1.30]; +1.10 [0.18, 1.99] | holds for all 3 seeds: +0.70; +3.25; +1.05 |
| F5 Transfer: D+A > D in mINP retention in all 15 cells (3.4) | – | holds for all 3 seeds (15/15 each) |
| F5 Transfer: D+A > D in absolute mINP in all 15 cells (3.4) | – | **15 / 13 / 15 cells**, see below |
| F6 Stage 2 multiplies NOMAD-val mINP over D+A by ≥ 2.5 (3.5) | 3.03× / 3.00× / 2.94× (NOMAD-val) | |

**12 of 13 checks hold for every seed.** The exception is narrow. For seed 1 on
RSTPReid, D+A does not beat D in absolute mINP in two cells: torso c = 0.1
(13.93 vs. 13.99) and torso c = 0.4 (9.71 vs. 9.87), differences of 0.06 and 0.16
points. In both cells D+A starts from a slightly lower clean mINP (14.85 vs. 15.21),
and in both it still retains a larger share of that clean mINP than D. The finding the
paper states for RSTPReid (better retention in all 15 cells, and higher mINP-AURC with
a CI excluding 0) holds for all three seeds. On CUHK-PEDES the same pattern occurs once
for seed 0, at torso c = 0.1 (41.33 vs. 41.38). The paper's claim there is made
against D+10, and it holds in 15 of 15 cells for every seed.

Seed spread of the headline quantities (mean ± SD over seeds 0, 1, 2):

| quantity | CUHK-PEDES | RSTPReid | NOMAD-val (Stage 2) |
|---|---|---|---|
| clean Rank-1 | 66.74 ± 0.48 | 48.62 ± 1.38 | – |
| clean mINP | 43.77 ± 0.61 | 14.87 ± 0.05 | 24.72 ± 0.37 |
| mINP-AURC | 66.23 ± 0.56 | 66.17 ± 0.75 | – |
| Δ mINP-AURC vs. D | +10.39 ± 0.56 | +9.55 ± 0.75 | – |
| Δ mINP-AURC vs. D+10 | +11.07 ± 0.56 | +8.88 ± 0.75 | – |
| foliage c = 0.5 mINP | 19.02 ± 0.78 | 6.22 ± 0.19 | – |

The seed-to-seed SD of the gain (0.56 to 0.75 points) is small next to the gain itself
(about 9 to 11 points). It is also of the same order as the half-width of a single
run's identity-bootstrap CI (0.7 points on CUHK-PEDES, 2.1 on RSTPReid).

Full per-seed tables: `results/reproduced/seed_consistency.md`.

## 5. Pending

* **ICFG-PEDES for D+A seeds 1 and 2.** These two models have not yet been scored on
  ICFG-PEDES. Commands (scoring only, no training):

  ```bash
  python tools/run_config.py eval configs/eval/orp_ground.yaml --model DA_s1 --benchmarks icfg --data-root ../Dataset --out-root runs
  python tools/run_config.py eval configs/eval/orp_ground.yaml --model DA_s2 --benchmarks icfg --data-root ../Dataset --out-root runs
  ```

  Once the per-query files exist (`tools/collect_per_query.py` picks them up
  automatically), `analysis/seed_consistency.py` adds ICFG-PEDES to every check above.
* **Reference runs are single-seed.** D, D+10, D+A′ and IRRA+A were trained once, so
  the D+A-vs-D+10 contrast above varies only the D+A seed.
