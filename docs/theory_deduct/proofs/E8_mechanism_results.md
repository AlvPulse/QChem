# E8: Place-then-Harvest Mechanism Probe

**Status:** COMPLETE (K=4, K=6, K=8 ALL DONE)  
**Script:** `run_levelG_probe.py --mechanism` (job blmupvrhb)  
**Theory connection:** T9 (place-then-harvest identity): entangler places signal into bonded
subspace S(K); readout harvests exactly those correlators.

---

## Protocol

For each K in {4, 6, 8} and each seed:
1. **PLACE probe**: train levelG structured model; extract mutual-information-like mass of
   circuit correlations on bonded vs. non-bonded pairs.
   - `on_m` = mean |ZZ| on true-bond pairs (A[i,j] > 0)
   - `off_m` = mean |ZZ| on non-bond pairs (A[i,j] = 0)
   - `onfrac` = fraction of total ZZ mass on bonded pairs
   - `base` = uniform expectation (fraction of K*(K-1)/2 pairs that are bonded)
2. **HARVEST probe**: measure readout alignment with true vs. random adjacency.
   - `h_true` = bond-pooled readout inner-product with true-A weights
   - `h_rand` = bond-pooled readout inner-product with random-A weights

T9 predicts:
- PLACE: on_m >> off_m (entangler preferentially gates onto bonds); onfrac >> base
- HARVEST: h_true >> h_rand (structured pool outperforms random pool)

---

## Results

### K=4 (COMPLETE)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| PLACE bonded (on_m) | 0.1560 | mean bond-pair ZZ mass |
| PLACE non-bonded (off_m) | 0.0287 | mean non-bond-pair ZZ mass |
| PLACE ratio | **5.43x** | bonded >> non-bonded |
| on-bond mass frac | 0.8305 | 83% of ZZ mass on bonds |
| uniform baseline (base) | 0.5078 | random expectation |
| HARVEST true-A (h_true) | 1.7545 | readout w/ true adj |
| HARVEST random-A (h_rand) | 1.0052 | readout w/ random adj |
| HARVEST ratio | **1.75x** | structured pool outperforms |

### K=6 (COMPLETE)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| PLACE bonded (on_m) | 0.0662 | mean bond-pair ZZ mass |
| PLACE non-bonded (off_m) | 0.0131 | mean non-bond-pair ZZ mass |
| PLACE ratio | **5.05x** | bonded >> non-bonded |
| on-bond mass frac | 0.7085 | 71% of ZZ mass on bonds |
| uniform baseline (base) | 0.3419 | random expectation |
| HARVEST true-A (h_true) | 2.2389 | readout w/ true adj |
| HARVEST random-A (h_rand) | 0.9814 | readout w/ random adj |
| HARVEST ratio | **2.28x** | structured pool outperforms |

**Note on K=6 vs K=4 onfrac:** The absolute onfrac drops from 0.830 to 0.709 because there
are more non-bonded pairs as K increases (K*(K-1)/2 grows faster than the number of bonds).
The baseline also drops from 0.508 to 0.342. The EXCESS onfrac vs baseline stays positive:
0.322 at K=4, 0.367 at K=6 -- so the PLACE signal is actually slightly STRONGER at K=6.

### K=8 (COMPLETE)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| PLACE bonded (on_m) | 0.0655 | mean bond-pair ZZ mass |
| PLACE non-bonded (off_m) | 0.0105 | mean non-bond-pair ZZ mass |
| PLACE ratio | **6.23x** | bonded >> non-bonded |
| on-bond mass frac | 0.6540 | 65% of ZZ mass on bonds |
| uniform baseline (base) | 0.2490 | random expectation |
| HARVEST true-A (h_true) | 2.918 | readout w/ true adj |
| HARVEST random-A (h_rand) | 0.989 | readout w/ random adj |
| HARVEST ratio | **2.95x** | structured pool outperforms |

---

## Interpretation

### PLACE confirmation

The graph-gated IsingXX entangler does exactly what T9 predicts: it concentrates quantum
correlations onto bond-pair qubits. At K=4 and K=6, the bonded/non-bonded ZZ ratio is
roughly 5x, far above the uniform null (1x). The on-bond mass fraction exceeds the baseline
by +32pp (K=4) and +37pp (K=6), with the excess GROWING with K.

This is NOT trivial: the entangler gates are parameterized by `adj[:, i, j] * pairp[l, pidx]`.
At random initialization, pairp is small but uniform. After training, the circuit has learned
to selectively amplify bond-pair entanglement. This is the "place" step of T9.

### HARVEST confirmation

The readout inner-product ratio h_true/h_rand is 1.75x at K=4 and 2.28x at K=6, and
increasing. This confirms T9's "harvest" step: bond-pooling with the true adjacency A
extracts more signal than bond-pooling with a random adjacency AR. The increasing ratio
(1.75 -> 2.28) is consistent with T9's SNR = alpha^2 * SNR_ideal scaling -- as K grows,
more bond-correlator mass is available to harvest.

### Cross-K summary (COMPLETE)

| K | PLACE ratio | HARVEST ratio | onfrac | base | onfrac excess |
|---|------------|---------------|--------|------|---------------|
| 4 | 5.43x | 1.75x | 0.830 | 0.508 | +0.322 |
| 6 | 5.05x | 2.28x | 0.709 | 0.342 | +0.367 |
| 8 | 6.23x | 2.95x | 0.654 | 0.249 | +0.405 |

Both probes confirm T9 operating end-to-end:
- PLACE ratio: nominally K-independent (~5.1-6.2x), rising at K=8 as the IsingXX entangler
  is more strongly focused on the sparser relative bond density
- HARVEST ratio GROWS monotonically with K: 1.75 -> 2.28 -> 2.95 (K=4,6,8)
- onfrac excess (alignment above uniform) GROWS with K: +0.322, +0.367, +0.405

The growing HARVEST ratio is the most direct confirmation of T9: as K increases, the
structured bond-pooled readout gains proportionally MORE relative to a random readout, exactly
because the bond subspace S(K) expands while containing more signal.

---

## T9 connection

**T9 (Place-then-Harvest identity):** alpha^2 * SNR_ideal is the signal recovered by the
bond-pooled readout, where alpha = (trained scale factor) and SNR_ideal = (zero-noise,
infinite-shot SNR in the bonded subspace).

The mechanism probe operationalizes this:
- PLACE = the "entangler places signal into bonded subspace S(K)" clause
- HARVEST = the "readout harvests exactly those correlators" clause

All three K values confirmed: T9 PASSES at K=4, K=6, and K=8.

---

## Files

- `results/mechanism_K4.npz`: K=4 raw arrays (on, off, on_m, off_m, onfrac, base, h_true, h_rand)
- `results/mechanism_K6.npz`: K=6 raw arrays
- `results/mechanism_K8.npz`: K=8 (DONE)
- `results/e8_mechanism_summary.json`: cross-K summary (K=4, K=8; K=6 from mechanism_K6.npz)
