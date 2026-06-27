# Inductive-Bias Probe: Does Graph-Topology Entanglement Carry Signal?

**Entry points**: `run_bias_probe.py` (experiment), `_verify_absorb.py` (control-validity proof).

This document records a debugging finding about `run_benchmark.py` and a clean re-test of the
project's central hypothesis.

---

## 1. The problem: the benchmark's scramble control is absorbable

`run_benchmark.py`'s headline is `structured − scrambled`: same circuit, same parameter
count, same depth; the "scramble" applies a *fixed random permutation* (`_perms` in
`src/quantum_levels.py`) that re-routes which feature column drives which wire. The claim is
that this destroys the chemistry→qubit correspondence and so isolates the inductive bias.

**It does not — because each modality is fed by a free `nn.Linear(64, n_qubits)` projection.**
Permuting the output columns of a free linear layer is identical to permuting its weight
rows, so training re-absorbs the permutation at zero cost. A single fixed permutation in
front of a free projection is a no-op for the optimizer.

### Proof (`_verify_absorb.py`, deterministic, no training) — all 7 levels
For each level we feed the structured circuit the raw inputs and the scrambled circuit the
correspondingly-permuted inputs (each input permuted by the first gate-perm applied to it),
with identical variational weights, and compare outputs. A residual of ~0 means a single input
permutation reproduces structured exactly, so the free upstream projection re-absorbs the
scramble and the control is vacuous.

| Level | residual | reuse structure | control status |
|---|---|---|---|
| 1 | — | no chemistry→operator routing | scramble ≡ structured — **sanity baseline, not a bias test** |
| **2** | **0.00e+00** | m, c, s: 1 perm each (own free proj) | **VACUOUS** (bit-exact at K=4, 6, 8) |
| 3 | 1.8e-01 | motif reused under 2 perms; c, s: 1 each | partial — only the motif reuse survives |
| **4** | **0.00e+00** | chem, dist: 1 perm each (both free proj) | **VACUOUS** |
| 5 | 1.3e+00 | chem reused under 4 perms (RZ,RY,XX,YY) | genuine |
| 6 | 1.6e+00 | chem reused under 5 perms | genuine |
| 7 | 1.4e+00 | chem reused under 5 perms (U3×3,CRX,CRY) | genuine |

A scramble is absorbable exactly when every input vector is used under a **single** permutation
(Levels 2, 4): the free projection then realises that permutation for free, so structured and
scrambled are the *same function class* and their delta is pure optimisation noise. The control
is genuine only where the **same** projected vector is reused under **multiple inconsistent**
permutations (Levels 5–7), partial where only one modality is reused (Level 3), and absent at
Level 1. So `run_benchmark.py`'s `structured − scrambled` headline is **uninterpretable at
Levels 1, 2, 4**, weak at Level 3, and only sound at Levels 5–7.

---

## 2. The clean re-test (`run_bias_probe.py`)

To test the hypothesis without the absorbability confound, the structured signal must live in
data that is **never routed through a free learnable layer**. We use the molecular graph
itself.

### Featurization
Each molecule is coarse-grained to **K spectral clusters** (K = n_qubits) over its bond graph.
Per cluster we keep 5 cheap features `[atomic_num, Gasteiger charge, degree, aromatic,
in_ring]`; cross-cluster bond weights form a **K×K coarse adjacency** `A`.

### Circuit (only the entangler differs across variants)
- Single-qubit encoding (identical in all variants): a tiny **fixed** `Linear(5, 2)` →
  `RY`, `RZ` per qubit. Deliberately minimal so nothing upstream can re-route topology.
- Entanglement: `IsingXX(A[i,j] · θ_pair)` on every qubit pair — the angle is gated by the
  **raw per-molecule adjacency `A`**, which is data, not a learnable projection, so it
  **cannot be absorbed**.

| Variant | Entangler | Isolates |
|---|---|---|
| `structured` | `IsingXX(true_A[i,j] · θ)` | the proposed bias |
| `scrambled` | `IsingXX(rand_A[i,j] · θ)`, same edge-weight multiset, shuffled topology | graph topology |
| `separable` | none (no IsingXX, no CRZ ring) | value of entanglement at all |
| `classical` | MLP on `[coarse feats ‖ adjacency]`, capacity-unconstrained | context baseline |

`scrambled` preserves edge **density** and the **multiset of bond weights** per molecule and
only shuffles *which pair* gets *which weight* — so `structured − scrambled` isolates the
value of the **correct topology**, with single-qubit information held identical.

### Evaluation (hardened)
- **Scaffold-grouped CV** (Bemis-Murcko, `GroupKFold`): structurally novel test folds (OOD),
  not a random split.
- **Validation-based epoch selection**: the reported epoch is chosen on a scaffold-disjoint
  validation ROC; the test fold is read once (no test-set peeking).
- **Primary test**: per-task paired Wilcoxon over the 12 Tox21 tasks on **pooled** CV
  predictions (each molecule predicted once), mirroring `run_benchmark.py`'s primary test;
  plus a sign test and run-level dAUC for reference.

---

## 3. Results

### 3a. Random-split sanity (15 seeds, K=4, paired)
Before the scaffold-CV hardening, a 15-seed paired random-split test (`_alt_b_multiseed.py`)
already showed a small but consistent signal once the confound was removed:

- structured > scrambled in **13/15 seeds**, mean dAUC **+0.0042**
- **sign test p = 0.0037**, **Wilcoxon p = 0.0062**, 95% CI [+0.0013, +0.0071]
- ordering matches theory: separable (0.664) < scrambled (0.672) < structured (0.673)

### 3b. Scaffold-CV qubit sweep (headline)
`run_bias_probe.py --qubits 4 6 8 --folds 3` (scaffold-grouped CV, pooled per-task paired
test; 2 seeds at K=4/6, 1 at K=8). `structured`/`scrambled` are pooled-CV across-seed means;
`classical` is a single-fold orientation value.

| K | pairs | structured | scrambled | median dAUC | tasks pos | sign p | Wilcoxon p | classical |
|---|---|---|---|---|---|---|---|---|
| **4** | 6 | 0.646 | 0.640 | **+0.0044** | 9/12 | 0.073 | **0.017** | 0.720 |
| 6 | 15 | 0.639 | 0.635 | +0.0020 | 8/12 | 0.194 | 0.170 | 0.682 |
| 8 | 28 | 0.655 | 0.655 | +0.0015 | 7/12 | 0.387 | 0.515 | 0.710 |

**The bias does not grow with entanglement capacity — it weakens.** It is significant at K=4
(Wilcoxon p=0.017, consistent with the random-split p=0.006 above), fades to non-significant
by K=6 (p=0.17), and is absent at K=8 (p=0.52 — and `separable` 0.670 actually *beats*
`structured` 0.655 there, i.e. entanglement of any topology stops helping). Classical leads
at every K by ~5–8 ROC points.

> **Power caveat.** K=6 used 2 seeds (which disagreed sharply: run-level +0.012 vs −0.003) and
> K=8 a single seed, so the higher-K cells are underpowered. The trend is "no evidence the
> bias grows with K", not a proof it is exactly zero there.

So **gate-gating alone produces a bias that does not scale** — which motivated a
measurement-based mechanism (Section 3c).

### 3c. Level G — measurement-based readout (the scalable mechanism)
`run_levelG_probe.py`. The bias is moved from gate routing into **which observables are read**,
selected by the graph: measure 2-qubit correlators ⟨Z_iZ_j⟩, ⟨X_iX_j⟩ for every pair, then
**bond-pool per qubit weighted by the coarse adjacency** — `b[i] = Σ_j A[i,j]·corr(i,j)`. The
adjacency multiplies the correlator *before* the head, so no free `Linear` can re-route which
correlation it harvests (non-absorbable); the readout is O(K) for sparse graphs and 2-local
Paulis are hardware-native (scalable). Three configs, each `structured` (true A) vs `scrambled`
(random A), scaffold CV, pooled per-task Wilcoxon:

| config | entangler | readout | K=4 (Wilcoxon p) | K=6 (Wilcoxon p) | K=8 (Wilcoxon p) |
|---|---|---|---|---|---|
| `gate` | graph-gated | single-qubit | +0.0044 (0.017) | +0.0026 (**0.13 n.s.**) | +0.0030 (**0.17 n.s.**) |
| `levelG` | graph-gated | + bond-correlators | +0.0078 (0.017) | **+0.0108 (0.011)** | **+0.0134 (0.0024)** |
| `meas_only` | fixed ring | bond-correlators | −0.027 (1.0) | — | — |

Two findings:
1. **Measurement readout makes the bias scale.** Gate-gating *fades* with K (K4 +0.0044 → K6
   +0.0026 n.s. → K8 +0.0030 n.s.); Level G *grows monotonically and gets MORE significant*
   (K4 +0.0078 p=0.017 → K6 +0.0108 p=0.011 → K8 **+0.0134 p=0.0024**, 10/12 tasks positive). The
   bias roughly doubles from K=4 to K=8 while the gate-only mechanism dies.
2. **The readout is an amplifier, not a standalone mechanism.** With a graph-independent ring
   entangler (`meas_only`), bond-pooling the *true* adjacency does *worse* than random
   (−0.027, 0/12 tasks positive): the correlations sit on ring edges, not molecular bonds, so
   harvesting them on bonds is a mismatch. The bias requires graph-gated entanglement to *place*
   correlations on the graph; the readout then harvests them.

---

## 4. Verdict

1. **A measurable quantum inductive bias exists, but only once the control is non-absorbable.**
   Structured beats scrambled at K=4 (Wilcoxon p=0.017 scaffold CV; p=0.006 random-split, 15
   seeds). `run_benchmark.py` cannot see it — its permutation scramble is re-absorbed by the
   upstream free projection, provably bit-exact at Levels 1/2/4 (Section 1).
2. **Gate-gating alone does not scale; measurement readout does.** The gate-gated bias weakens
   with qubits (n.s. by K=6). Routing the bias through graph-selected 2-qubit correlators
   (Level G) makes it *grow* with K and stay significant (p=0.0105 at K=6) — the scalable,
   non-absorbable, hardware-native mechanism. The readout only works *with* graph-gated
   entanglement (it amplifies, it does not create).
3. **Still not competitive with classical** at this scale — a capacity-unconstrained MLP on the
   same coarse features leads by ~5–8 ROC points. The contribution is a *correct, scalable
   isolation of the inductive bias*, not a performance claim.

---

## 5. What "fixing `run_benchmark.py`" actually requires (architectural root cause)

Every quantum layer in the benchmark receives its inputs as `extractor(data)` (a trainable GNN)
followed by a trainable `nn.Linear` projection — including Level 4's 3D distances, which go
through `dist_proj`. **No fixed per-molecule data reaches any circuit.** A scramble that permutes
the projection outputs is therefore always downstream of a free trainable map that can
re-compensate. There are only two ways to make a scramble non-absorbable:

- **(a) Reuse** — feed the *same* projected vector into multiple gates under *conflicting*
  permutations (what Levels 5–7 already do, which is why their controls are genuine). Making
  Levels 2/4 non-absorbable this way means restructuring them to reuse a shared vector — at
  which point they become mechanistically the same test as 5–7.
- **(b) Fixed data** — inject a non-trainable per-molecule structure that the scramble permutes
  (the probe's raw coarse adjacency). This requires the **qubits-as-graph-nodes** featurization,
  not the benchmark's qubits-as-abstract-feature-carriers design.

Consequence: the benchmark's seven levels treat qubits as abstract feature slots projected from a
molecule vector; they have **no native molecular-graph structure** for a topology bias to live in.
Level G (option b) is the architecturally correct home for a non-absorbable, scalable bias — and
it is already implemented in the probe (`run_bias_probe.py` / `run_levelG_probe.py`). Patching
Levels 2/4 via option (a) would make their controls valid but redundant with 5–7 and would not
add a scalable graph bias. This is the key decision for productionization (Section 4 of the
project plan).
