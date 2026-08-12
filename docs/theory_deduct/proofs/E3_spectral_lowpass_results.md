# E3: Spectral Low-Pass Verification Results
*Tests T3 Lemma 3.5: spectral coarsening retains low-graph-frequency feature content*
*Date: 2026-07-12 | Script: probe_spectral_lowpass.py | N=200 molecules, Tox21+ToxCast*

---

## Results

```
   K | n_valid | frac_low(K/2) |      gap |  disc_err | interp
--------------------------------------------------------------
   4 |     200 |        0.9344 |   0.7038 |    0.2107 | low-pass OK
   6 |     200 |        0.9302 |   0.4115 |    0.2063 | low-pass OK
   8 |     200 |        0.9238 |   0.2269 |    0.2196 | low-pass OK
```

**Metric definitions:**
- `frac_low(K/2)`: fraction of coarse feature energy in the bottom K/2 Laplacian eigenmodes.
  Pass criterion: > 0.5 (most energy in low-frequency band). T3 STRONGLY PASSED.
- `gap`: lambda_{K/2+1} - lambda_{K/2} (spectral gap at midpoint). Larger = cleaner cluster separation.
- `disc_err`: ||qf_coarse - Pi_{K/2} qf|| / ||qf_coarse|| (discretization residual).

---

## T3 Validation Assessment

**PASS with margin.** At all K=4,6,8, over 93% of the coarse feature energy sits in the
bottom-K/2 Laplacian eigenspace of the coarse graph. This strongly confirms T3 Lemma 3.5
(spectral coarsening = approximate ideal low-pass filter).

### Key findings

**1. Low-frequency fraction: ~93% across all K**
The frac_low values 0.934, 0.930, 0.924 are remarkably stable across K=4,6,8.
This means: regardless of how many clusters are used, the CLUSTER-MEAN operation
produces node signals that sit overwhelmingly in the smooth (low-graph-frequency) subspace.
The T3 relaxation claim (ideal low-pass) is tightly satisfied.

**2. Spectral gap decreases with K: 0.70 -> 0.41 -> 0.23**
More clusters = smaller inter-cluster spectral gap (more closely-spaced eigenvalues).
This is expected: for K clusters on a molecule with ~20-40 atoms, lambda_{K+1} - lambda_K
shrinks as K approaches the molecule size. At K=8, gap=0.23 is still substantial,
meaning the clusters remain well-separated at this K. This gap sets the Davis-Kahan
bound on discretization error (T3, Section 3b).

**3. Discretization error: ~21% at all K**
disc_err ~= 0.21 means the 'discretize' rounding introduces ~21% residual relative to
the ideal projection Pi_K. This is the HONEST SCOPE of T3 Part (b): the implementation
is not bit-exact ideal low-pass -- it has a 21% discretization perturbation. However,
93% low-frequency fraction shows that even with this perturbation, the output is
predominantly low-graph-frequency.

**4. Stability of frac_low across K**
The decreasing gap (0.70 -> 0.23) does NOT lead to decreased low-pass quality
(frac_low stays 0.93). This is because: as K increases, the cluster-mean pooling
naturally averages over smaller groups, which still produces smooth signals. The
spectral gap controls when clusters MERGE (bad: wrong assignment), not whether
CLUSTER MEANS are smooth.

---

## Implication for the TC-QIC double bottleneck

T3 Lemma 3.5 states the FIRST bottleneck is the topological one: the cluster-mean
operation discards the high-frequency atomic noise. E3 quantifies this:

- 93% of feature energy is in the low-frequency band -> the topological bottleneck
  retains the smooth (cluster-level) molecular structure.
- 7% is high-frequency -> this is the epsilon_spectral in T7 Corollary 5
  (the information loss from coarse-graining).
- disc_err = 21% -> the practical implementation deviates from ideal by this factor,
  but the resulting features still satisfy the low-pass criterion.

This is consistent with Cor 3.6: tasks that are low-graph-frequency (toxicophore location
= cluster-level pattern) should have I(C(G);Y) ~= I(G;Y) (small epsilon_spectral).
Tasks that are high-graph-frequency (atom-level distinctions) will show AUC ceilings
from the spectral bottleneck -- this is what E4 will measure.

---

## Status: E3 DONE -- T3 Lemma 3.5 EMPIRICALLY CONFIRMED (93% low-frequency fraction)
Next: E4 (epsilon-sufficiency / info preservation test) -- measures the 7% high-frequency loss.
