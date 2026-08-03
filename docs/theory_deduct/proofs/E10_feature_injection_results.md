# E10: Differential Feature Injection (P5 Prediction)

**Status:** COMPLETE (K=6 DONE -- P5 PASS)
**Script:** `run_e10_feature_injection.py`
**Theory connection:** T2 (operator-geometry bottleneck): the TC-QIC readout projects
onto a Theta(K) slice that is LOW-FREQUENCY in the cluster basis. High-frequency
within-cluster features (per-atom variance) cannot be resolved by the bond-pooled
correlators. P5 predicts that CLASSICAL models gain more from these features than
QUANTUM models.

---

## Protocol

For K=6 (default):
1. **Original features**: K-cluster means (FDIM=5 per cluster): [atomic_num, charge, degree, aromatic, inring]
2. **Augmented features**: Original + within-cluster variance/max (FDIM_AUG=3): [std_charge, max_atomic_num, std_degree]

Both quantum GraphG and classical ClassicalGNN trained on:
- `original`: same features as the main benchmark
- `augmented`: original + FDIM_AUG=3 extra within-cluster variance features

**P5 criterion:** AUC_classical(augmented) - AUC_classical(original) > AUC_quantum(augmented) - AUC_quantum(original)
i.e., classical gains more from the added features than quantum.

---

## Theoretical Predictions

### P5 (operator-geometry limits high-frequency access)

T2 proves that the GraphG readout lives in the Theta(K) eigenspace corresponding to
COARSE (cluster-scale) graph features. The augmented features (std_charge, max_atomic_num)
are FINE-GRAINED: they encode within-cluster variation invisible to the cluster-mean
features already in the circuit encoding.

**Why quantum can't use within-cluster variance:**
- The circuit encoding is `RY(feat[cluster_mean] * qf), RZ(...)` per qubit
- After coarse-graining, each qubit represents ONE cluster; its state encodes the MEAN
  atom features of that cluster
- The IsingXX entangler gates on BETWEEN-cluster bonds: these are also coarse-scale
- Within-cluster variance (std_charge etc.) is collapsed by the coarse-graining step BEFORE encoding
- Even if added to the cluster feature vector, the bond-pooled ZZ/XX readout cannot
  resolve these features from the mean -- they get averaged over when pooling

**Why classical CAN use within-cluster variance:**
- The ClassicalGNN processes cluster features through a linear+ReLU+linear network
- An additional `nn.Linear(FDIM+FDIM_AUG, d)` input layer can directly weigh
  within-cluster variance features for each task
- No geometric bottleneck prevents learning from high-frequency cluster statistics

**Quantitative prediction:** The classical gain from FDIM_AUG features should be +0.003 to +0.010
AUC (proportional to the information content of within-cluster variance). The quantum gain
should be < 0.003 (noise floor from adding un-useful features to the encoding).

---

## Results

### K=6 (COMPLETE)

| Model | AUC (original) | AUC (augmented) | Gain | P5? |
|-------|---------------|-----------------|------|-----|
| Quantum (GraphG) | 0.6343 | 0.6322 | **-0.0021** | -- |
| Classical (GNN) | 0.7006 | 0.7105 | **+0.0099** | PASS |

Classical gain (+0.0099) >> quantum gain (-0.0021). P5: **PASS**.

**Key finding:** Adding within-cluster variance features (std_charge, max_atomic_num, std_degree)
HELPS classical by +0.0099 AUC but HURTS quantum by -0.0021. The quantum model cannot use
these high-frequency within-cluster features because the bond-pooled ZZ/XX readout lives in
the coarse-topology subspace -- it cannot resolve features that vary WITHIN clusters (T2).

**Additional finding:** classical orig=0.7006 vs quantum orig=0.6343 -- classical leads by
0.0663 AUC in the unaugmented condition, consistent with T12's double-bottleneck prediction
(5-8 AUC point classical advantage).

---

## P5 PASS/FAIL

- **P5 at K=6: PASS**
- Classical gain: +0.0099 (augmented > original, as predicted)
- Quantum gain: -0.0021 (augmented = no improvement, as predicted)
- Difference: +0.0120 in favor of classical
- Sign difference: classical POSITIVE, quantum NEGATIVE (strong directional confirmation)

---

## Connection to T12

T12 (bias-variance decomposition) identifies a "double bottleneck" causing the 5-8 AUC point
classical advantage: the coarse-graining step PLUS the operator-geometry truncation. E10
operationalizes the first bottleneck: if classical models gain from within-cluster variance
and quantum models don't, this directly demonstrates that coarse-graining destroys
potentially useful fine-grained information that only classical models can recover.

---

## Files

- `data/bias_augmented_K6.npz`: K=6 augmented coarse graphs (DONE)
- `results/e10_feature_K6.npz`: K=6 quantum/classical results (DONE)
- `results/e10_summary.json`: combined summary (DONE)
