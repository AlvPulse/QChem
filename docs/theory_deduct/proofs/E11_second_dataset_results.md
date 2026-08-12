# E11: Second-Dataset External Validity (P7 Prediction)

**Status:** PARTIAL (K=4 DONE, K=6 DONE, K=8 running -- job b1tlx5vof)
**Script:** `run_e11_second_dataset.py`
**Theory connection:** P7 (slope invariance): the TC-QIC normalized K-scaling slope
(d_K Delta / d_bar) should be DATASET-INVARIANT, confirming that the bias is driven
by graph topology, not by idiosyncrasies of the Tox21/ToxCast chemical series.

---

## Protocol

Dataset: BBBP (Blood-Brain Barrier Permeability, n=1975, 1 binary task)
K=4,6,8 levelG structured vs scrambled, 3 scaffold folds, 1 seed, 20 epochs.

**P7 criterion:**
- Compute K-slope for BBBP: slope_BBBP = (dAUC(K=8) - dAUC(K=4)) / (8-4)
- Normalize by mean graph degree: norm_slope = slope / d_bar
- Tox21 reference slope: 0.0028 dAUC/qubit = (0.0134-0.0078)/(8-4)
- P7 PASS if |slope_BBBP / slope_Tox21| < 3x AND K-rank order preserved

**Additional test:**
- Kendall tau: check that dAUC(K=4) < dAUC(K=6) < dAUC(K=8) for BBBP
  (same rank order as Tox21 confirms topology-driven mechanism, not dataset artifact)

---

## Theoretical Predictions

### P7 (normalized slope invariance)

If the TC-QIC bias is driven by GRAPH TOPOLOGY (T1-T10), then the per-qubit dAUC gain
should be controlled primarily by the coarse-graph structure (K, bond density, spectral gap)
rather than by chemical task specifics (toxicity vs. BBB permeability).

The normalization by d_bar (mean graph degree) accounts for molecular complexity:
- BBBP molecules: drug-like, moderate complexity (~30 heavy atoms)
- Tox21 molecules: diverse (environmental chemicals to drugs)

If both datasets give similar normalized slopes, it supports that K-scaling is
architecture-driven (T10) rather than dataset-specific.

**Tox21 reference:** slope = 0.0028 dAUC/qubit, d_bar ~ 2.0 (for K=6 coarse graphs)
**BBBP expected:** similar normalized slope (within 3x factor, given different task/dataset)

### Why might slopes differ?

If BBBP has different bond density (d_bar) than Tox21, the absolute slope could differ
even if the mechanism is the same. The 3x tolerance accounts for this. If the ratio
exceeds 3x, it suggests the scaling is dataset-specific (P7 FAIL).

---

## Results

### BBBP K=4,6,8 (IN PROGRESS -- job b1tlx5vof)

| K | struct AUC | scram AUC | dAUC | d_bar | norm dAUC | K-slope |
|---|-----------|----------|------|-------|-----------|---------|
| 4 | 0.6576 | 0.6243 | +0.0333 | 1.66 | +0.0201 | baseline |
| 6 | 0.6612 | 0.6566 | +0.0046 | 2.01 | +0.0023 | -0.01435/qubit |
| 8 | *running* | | | | | |

**K=4 BBBP:** dAUC = +0.0333, much larger than Tox21 K=4 (+0.0078).
**K=6 BBBP:** dAUC = +0.0046 -- SHARP DROP from K=4. K-slope K=4->6 = (0.0046-0.0333)/2 = -0.01435/qubit (NEGATIVE).

**Critical observation:** The BBBP K-rank is NOT preserved at K=4,6 -- dAUC(K=4) >> dAUC(K=6).
This is opposite to Tox21 where dAUC increases monotonically with K.

**Interpretations to consider:**
1. BBBP K=4 may be an outlier -- 1975 molecules, 1 task, high variance in scaffold CV
   (only ~440 test molecules per fold vs Tox21's ~2600+)
2. Drug-like molecules may have optimal topology resolution at K=4 (~7-8 atoms/cluster),
   where the coarse-graph captures the key scaffold-level features (ring systems, chains)
3. At K=6 (d_bar=2.01 vs K=4 d_bar=1.66), clusters become smaller and the topology
   may no longer align with the chemically relevant substructures
4. Single-task BBBP (1 task AUC vs Tox21 Wilcoxon over 12 tasks) has higher variance

**P7 implications:** K=8 result is critical. If dAUC(K=8) > dAUC(K=6), K-rank is partially
restored for K=6->8, even if K=4 is anomalously high. Full P7 assessment after K=8 complete.

### Tox21 reference (COMPLETE -- main benchmark)

| K | median dAUC (scaffold CV) | d_bar |
|---|--------------------------|-------|
| 4 | +0.0078 | ~2.0 |
| 6 | +0.0108 | ~2.0 |
| 8 | +0.0134 | ~2.0 |

Tox21 slope: (0.0134 - 0.0078) / (8 - 4) = 0.0014 dAUC/qubit
Note: slope over K=4-8 range, not per-qubit rate from T10 fit (which is 0.0014).

---

## P7 PASS/FAIL (partial)

**Tox21 reference slope:** 0.0014 dAUC/qubit (K=4->8 range)

**BBBP (K=4,6 done):**
- K-slope K=4->6: (0.0046 - 0.0333)/2 = **-0.01435/qubit** (NEGATIVE)
- K-rank at K=4,6: NOT preserved (dAUC(K=4) >> dAUC(K=6), reversed)
- Slope ratio vs Tox21: -0.01435 / 0.0014 = -10.25x (SIGN REVERSED, |ratio| >> 3x)

**Status as of K=4,6:** P7 AT RISK -- slope sign reversed between Tox21 and BBBP.

**Pending K=8:**
- If dAUC(K=8) > dAUC(K=6) = 0.0046: K-rank K=6->8 restored (partial)
- For overall K=4->8 slope to be positive: need dAUC(K=8) > 0.0333 (unlikely given K=6=0.0046)
- Most likely: K=4->8 slope will remain negative for BBBP

**P7 likely outcome:** FAIL for strict slope-invariance criterion.
The K-scaling is dataset-dependent: Tox21 shows monotone increase, BBBP shows peak at K=4.

**Alternative framing (honest):** TC-QIC bias depends on the match between K (topology
resolution) and the chemically relevant scale for the task. BBBP (drug-like, BBB
permeability) peaks at K=4 coarse-graphing; Tox21 (diverse chemicals, 12 tasks) benefits
from finer K. This is consistent with the T10 mechanism (bond-correlator signal grows with
K) but the absolute dAUC also depends on task-topology alignment, which is dataset-specific.

---

## Files

- `data/featurized_BBBP.pt`: raw BBBP graph dataset (DONE, 1975 molecules)
- `data/bias_BBBP_K{k}.npz`: K=4,6,8 coarse graphs (will be created by E11)
- `results/e11_kscale_BBBP.json`: K=4,6,8 dAUC results (pending)
