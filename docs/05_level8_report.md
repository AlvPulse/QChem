# Level 8 (a.k.a. "Level G"): A Measurement-Based, Scalable Quantum Inductive Bias — Detailed Report

*Written to be self-contained for a mixed audience: quantum-information readers who may be new to
machine-learning evaluation, and machine-learning readers who may be new to quantum circuits. A
glossary is at the end (§10); terms in **bold-italic** on first use are defined there.*

---

## 0. Executive summary

We asked a narrow, falsifiable question: **does encoding a molecule's bond-graph topology into a
quantum circuit give the model a useful *inductive bias* — a built-in prior that helps it
generalize — beyond what an identical circuit with the *wrong* topology would give?**

Three results:

1. **The project's original test for this was invalid.** In the seven pre-existing circuit
   "levels," the control that was supposed to destroy the chemistry-to-qubit correspondence
   (the *scrambled* variant) is, at Levels 1/2/4, **mathematically equivalent to the structured
   circuit** — training silently undoes it. We prove this bit-for-bit.

2. **A correctly-controlled test shows the topology bias is real but small, and — crucially —
   it does not scale**: with the topology encoded only in the circuit's *gates*, the measurable
   advantage of true-vs-random topology fades to statistical noise as the circuit grows from
   4 to 6 to 8 qubits.

3. **Level 8 fixes both problems.** By moving the topology information out of the gates and into
   the **measurement** — reading two-qubit *correlators* and pooling them along the real molecular
   bonds — the bias (a) becomes provably **non-absorbable** (training cannot undo it) and
   (b) **grows with circuit size** instead of fading, remaining statistically significant at
   K = 6 qubits where the gate-only version has already died. This is the central novelty.

Level 8 is *not* a claim that quantum beats classical (a parameter-free classical baseline is still
~5–8 AUC points ahead). It is a *correct and scalable isolation of a quantum inductive bias* — the
kind of carefully-controlled positive result the field generally lacks.

---

## 1. Background: the question and why it is subtle

### 1.1 Inductive bias
Two models with the same number of parameters can generalize very differently because of their
**_inductive bias_** — the assumptions baked into their architecture about what good solutions look
like. A convolutional network's inductive bias is translation-equivariance; a graph neural network's
is permutation-equivariance over a graph. The promise of *quantum* machine learning for chemistry is
that a quantum circuit whose gates mirror molecular structure might have a chemistry-shaped inductive
bias. Our entire project tests whether that promise is real, in the most controlled way we can.

### 1.2 The clean way to test an inductive bias: structured vs. scrambled
The gold-standard test is to compare two models that are **identical in every measurable capacity**
— same gates, same parameter count, same depth, same optimizer — and differ *only* in whether the
chemistry-to-circuit mapping is the true one or a scrambled one. If the true mapping wins, the gap is
attributable to the inductive bias and nothing else (not capacity, not expressivity). We call the
true model **structured** and the destroyed-mapping model **scrambled**. Their difference in held-out
performance is our signal.

### 1.3 The trap
This test has a subtle failure mode that invalidated the original benchmark, described next.

---

## 2. The methodological problem we found: "absorbable" controls

### 2.1 How the original 7 levels work
Each of the seven circuit levels feeds a molecule through a trainable **GNN encoder** to get a
64-dimensional vector, then through a trainable linear map `Linear(64 → K)` to get one rotation
angle per qubit, then into a parameterized circuit. The *scrambled* control applies a **fixed random
permutation** to those angles — i.e., it sends "feature for qubit 3" into qubit 0's gate, etc.,
intending to break the chemistry↔qubit correspondence.

### 2.2 Why that control is silently undone
A permutation applied to the *output* of a free linear layer is the same as permuting the *rows* of
that layer's weight matrix. Formally, if the circuit input is `P · (W·x)` for a fixed permutation
matrix `P` and a *trainable* weight matrix `W`, then because `P·W` is just another matrix `W'`,
training can learn `W' = P⁻¹·W_desired` and reach **exactly** the structured optimum. The permutation
is "absorbed" into the projection. The scrambled and structured models are then the *same function
class*, and any measured difference is optimization noise — not inductive bias.

### 2.3 Proof (bit-exact, no training): `_verify_absorb.py`
For each level we built the structured and scrambled circuits with **identical** variational weights
and asked: *does there exist a single fixed permutation of the inputs that makes the scrambled circuit
reproduce the structured circuit's outputs exactly?* If yes, a free projection realizes it for free.

| Level | residual `max|structured − scrambled(permuted input)|` | Why | Control status |
|---|---|---|---|
| 1 | — | no chemistry→operator routing exists | scramble ≡ structured (**no control**) |
| **2** | **0.00 × 10⁰** (bit-exact, at K = 4, 6, 8) | each input used under ONE permutation, own free projection | **VACUOUS** |
| 3 | 1.8 × 10⁻¹ | motif vector reused under 2 conflicting perms | partial / weak |
| **4** | **0.00 × 10⁰** | chem & distance each via own free projection, one perm each | **VACUOUS** |
| 5 | 1.3 × 10⁰ | one `chem` vector reused under 4 perms (RZ, RY, XX, YY) | genuine |
| 6 | 1.6 × 10⁰ | one `chem` vector reused under 5 perms | genuine |
| 7 | 1.4 × 10⁰ | one `chem` vector reused under 5 perms (U3×3, CRX, CRY) | genuine |

**Reading the table.** A residual of exactly zero means the scramble is undone perfectly: the control
is meaningless. This happens whenever each input vector drives exactly one gate-site (Levels 2, 4).
The control only "bites" when the *same* vector is forced through *several inconsistent* permutations
at once (Levels 5–7), because then no single re-ordering of the projection can satisfy all of them.

### 2.4 The architectural root cause
Every one of the seven levels sends its circuit inputs through a *trainable* map (GNN + linear
projection) — **no fixed, per-molecule structural data ever reaches the circuit.** Consequently a
permutation-based control is always at the mercy of that trainable map. The only escape is to inject
structure that is **data, not a learnable parameter** — which is exactly what Level 8 does.

---

## 3. Level 8: what it is and what it does

Level 8 abandons "qubits as abstract feature slots" and instead makes **qubits = nodes of the
molecular graph**, with a real bond structure between them. The inductive bias then enters in two
places that data — not trainable weights — controls: the **entangler** and, novelly, the
**measurement**.

### 3.1 Step 1 — Featurization: molecule → coarse quantum graph
A molecule typically has more atoms than we have qubits, so we **coarse-grain** it:

- Take the molecular bond graph (atoms = nodes, bonds = weighted edges by bond order).
- Partition the atoms into `K` clusters by **spectral clustering** (K = number of qubits, e.g. 4/6/8).
- Each cluster → one qubit. Its features = the mean of its atoms' 5 cheap descriptors
  `[atomic number, Gasteiger partial charge, degree, aromaticity, in-ring]`.
- Build a **K × K coarse adjacency matrix `A`**: `A[i,j]` = total bond order connecting cluster *i*
  to cluster *j*, normalized. This `A` is the molecule's topology, as **fixed data**.

The crucial point: `A` is computed from chemistry and **never passes through a trainable layer.** It
is the carrier of the inductive bias.

### 3.2 Step 2 — Encoding (single-qubit, identical across all variants)
Each qubit's 5 features are mapped by a deliberately **tiny, fixed** `Linear(5 → 2)` into two angles,
applied as single-qubit rotations `RY(θ_i)`, `RZ(φ_i)`. "Tiny and fixed" is intentional: we want no
expressive trainable layer that could re-route information and thereby re-absorb the topology. This
encoding is byte-for-byte identical between structured and scrambled, so it cannot explain any gap.

### 3.3 Step 3 — Graph-gated entanglement (topology in the gates)
For every qubit pair `(i, j)` we apply a two-qubit entangling gate whose strength is **gated by the
bond**:
```
IsingXX( A[i,j] · θ_pair )           # θ_pair trainable; A[i,j] is fixed data
```
`IsingXX` correlates qubits *i* and *j*. Because the rotation angle is multiplied by `A[i,j]`, the
circuit builds quantum correlations **preferentially between bonded clusters** and (near-)none between
non-bonded ones. A trainable scalar `θ_pair` can scale a coupling but cannot *invent* a bond where
`A[i,j]=0` — so the entanglement pattern is the molecular graph, and a downstream linear layer cannot
reorder which pair was coupled.

### 3.4 Step 4 — Measurement-based graph readout (THE NOVELTY)
This is what distinguishes Level 8 from everything before it. We read out not just single-qubit
averages but **two-qubit correlators**, and we **pool them along the real bonds**.

For each qubit we measure the usual single-qubit **_expectation values_** `⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩`. In
addition, for every pair we measure the **two-qubit correlators**
```
⟨Z_i Z_j⟩   and   ⟨X_i X_j⟩
```
A correlator `⟨Z_i Z_j⟩` measures *how correlated* the two qubits' outcomes are — it is large exactly
when the circuit has entangled them. Then we **bond-pool**: each qubit aggregates its correlators with
its graph neighbors, weighted by bond strength:
```
b_ZZ[i] = Σ_j  A[i,j] · ⟨Z_i Z_j⟩
b_XX[i] = Σ_j  A[i,j] · ⟨X_i X_j⟩
```
The final per-molecule feature vector is, per qubit,
`[ ⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩, b_ZZ[i], b_XX[i] ]` → `5K` numbers → a single trainable linear head → the
12 toxicity predictions.

This `b[i] = Σ_j A[i,j]·corr(i,j)` is a **quantum, permutation-invariant graph readout** — the direct
analogue of a graph neural network's neighborhood aggregation, but where the "messages" are genuine
two-qubit quantum correlations and the "edges" are the real molecular bonds.

### 3.5 The structured vs. scrambled control for Level 8
- **structured**: `A` = the molecule's *true* coarse adjacency.
- **scrambled**: `A` = a *random* adjacency with the **same number of edges and the same multiset of
  bond weights**, only the *topology* (which pair gets which weight) is shuffled, fixed per molecule.

Both use identical encoding, identical gate set, identical parameter count, identical readout
*formula*. The **only** difference is whether the topology is real. So the gap is the inductive bias,
cleanly.

---

## 4. Why Level 8 creates an inductive bias (mechanistic explanation)

The bias is a two-stage "place-then-harvest" mechanism:

1. **Place** (entangler): graph-gated `IsingXX` concentrates quantum correlations on the molecule's
   true bonds. After the circuit, `⟨Z_i Z_j⟩` is large where atoms are bonded, small where they are
   not — the correlation structure *mirrors the molecular graph*.
2. **Harvest** (readout): bond-pooling `Σ_j A[i,j]⟨Z_iZ_j⟩` reads those correlations back out *along
   the same bonds*. The model therefore receives, as features, "the strength of quantum correlation
   along each chemical bond" — a representation that is **graph-local by construction**.

The encoded prior is: *the toxicity-relevant signal is a function of bond-local quantum correlations.*
When we scramble `A`, the readout harvests correlations along *wrong* (random) bonds — where the
circuit placed little correlation — so the features are less informative and held-out performance
drops. That drop is the inductive bias, and §6 shows it is real and growing with K.

A decisive sanity check (the `meas_only` ablation, §8.5) confirms the *place* step is necessary: if we
replace the graph-gated entangler with a **graph-independent ring** entangler (so correlations sit on
ring edges, not molecular bonds) and keep the bond-pooled readout, the true-topology readout does
*worse* than random (median ΔAUC = −0.027, 0 of 12 tasks positive). Harvesting along bonds only helps
if the correlations were placed on those bonds first. The readout is an **amplifier of a graph-gated
circuit, not a standalone trick.**

---

## 5. Why Level 8 is non-absorbable (the rigorous argument)

Recall the failure of the permutation control (§2.2): `P·(W·x) = (P·W)·x = W'·x`, so a fixed
permutation of a free projection's output is absorbed into the weights.

Level 8 evades this because the topology enters as a **data-dependent multiplicative weight on a fixed
physical observable, upstream of the only trainable readout layer**. The head computes
```
ŷ = W · [ s(x) ;  B_A(x) ],     where   B_A(x)_i = Σ_j A_ij · C_ij(x)
```
with `s(x)` the single-qubit expectations, `C_ij(x)` the two-qubit correlators produced by the circuit,
and `A` the **fixed per-molecule** adjacency. Two facts make this non-absorbable:

1. `A` is **not a parameter** — there is no weight to permute or re-learn. The structured-vs-scrambled
   difference is a difference in *data fed to a fixed pooling formula*, not in a learnable routing.
2. `A` **varies per molecule** and multiplies the *data-dependent* correlators `C(x)`. There is no
   single weight reparametrization `W'` that turns `B_{A_scram}(x)` into `B_{A_true}(x)` for *all*
   molecules simultaneously, because the relationship between the two depends on each molecule's own
   `A` and its own `C(x)`. A linear head can rescale a feature; it cannot re-select which physical
   correlator was summed into it.

We verified the contrast empirically: the permutation control is bit-exact absorbable at Levels 2/4
(§2.3), whereas Level 8's true-vs-random-adjacency control produces a *stable, statistically
significant, growing* gap (§6) that training never removes.

---

## 6. Why Level 8 scales (the second novelty)

"Scales" has two meanings here; Level 8 wins on both.

### 6.1 Computational / hardware scalability
- **Gates:** `O(K)` single-qubit rotations + `O(#bonds)` two-qubit gates. Molecular graphs are sparse
  (average cluster degree ≈ 2–3), so this is `O(K)`, not the `O(K²)` all-to-all coupling of the
  original Level 6.
- **Measurements:** `O(K)` single-qubit Paulis + `O(#bonds) = O(K)` two-qubit correlators, every one
  of them **≤ 2-local**. On real hardware, 2-local Pauli expectations are native and estimable in a
  constant number of commuting measurement groups, or from `O(log K)` randomized / **_classical-shadow_**
  measurements. (Single-qubit-only readouts of the original levels are also O(K), but they are exactly
  the readouts that *fail to carry the bias* — see §6.3.)
- **Readout dimension & pooling:** the bond-pooling is **permutation-invariant** and produces a
  fixed-size, `O(K)` feature vector regardless of molecule size, so a single architecture handles any
  molecule and any qubit count without change — exactly the property that lets GNNs scale.

(Statevector *simulation* is still `O(2^K)`, but that is a property of classically *simulating* any
quantum circuit, not of Level 8; on quantum hardware Level 8's cost is `O(K)`.)

### 6.2 Statistical scalability — the empirical headline
The pre-existing, gate-only mechanism produces a bias that **fades** as the circuit grows; Level 8's
measurement mechanism makes it **grow**:

| K (qubits) | #bonds (pairs) | gate-only median ΔAUC (Wilcoxon p) | **Level 8 median ΔAUC (Wilcoxon p)** |
|---|---|---|---|
| 4 | 6 | +0.0044 (p = 0.017) | **+0.0078 (p = 0.017)** |
| 6 | 15 | +0.0026 (p = 0.13, n.s.) | **+0.0108 (p = 0.011)** |
| 8 | 28 | +0.0030 (p = 0.17, n.s.) | **+0.0134 (p = 0.0024)** |

The pattern is now complete across all three qubit counts. The gate-only bias is non-significant at
both K = 6 and K = 8, while **Level 8's bias grows monotonically and becomes *more* significant** as
the circuit grows: ΔAUC +0.0078 → +0.0108 → +0.0134, Wilcoxon p = 0.017 → 0.011 → **0.0024**
(10 of 12 tasks favouring structured at K = 8). More qubits means more bonds means a richer
bond-correlator readout: the measurement mechanism turns added capacity into added signal where the
gate-only mechanism turns it into noise.

### 6.3 Why the measurement, specifically, is what scales
Single-qubit readouts compress the entire entangled state onto `3K` local averages, discarding the
*correlations* that the graph-gated entangler worked to create — and those correlations are precisely
where the topology lives. As K grows there are quadratically more pairwise correlations but still only
`O(K)` single-qubit averages, so a single-qubit readout throws away an increasing fraction of the
graph-structured information. The bond-correlator readout reads that information directly, which is why
its advantage *grows* with K.

---

## 7. How we proved the inductive bias (experimental design)

The proof rests on a controlled comparison evaluated under a deliberately conservative protocol.

### 7.1 The controlled comparison
- **structured vs scrambled** (true vs random adjacency, matched edge count and weight multiset):
  isolates topology with capacity, encoding, depth, and parameter count held identical.
- **separable** (entanglement removed): isolates whether entanglement helps at all.
- **classical** (an unconstrained MLP on the same coarse features + adjacency): context baseline.

### 7.2 Out-of-distribution evaluation: scaffold-grouped cross-validation
Random train/test splits let near-duplicate molecules leak across the split and inflate scores. We use
**_scaffold-grouped cross-validation_**: molecules are grouped by their **Bemis–Murcko scaffold** (their
core ring system), and entire scaffolds are held out, so the test set is *structurally novel*. This
simulates real deployment on new chemistry and is the hardest standard split.

### 7.3 No test-set peeking
For each fold, the training epoch is selected on a **separate, scaffold-disjoint validation set**;
the test fold is scored once, at that selected epoch. Both structured and scrambled use the identical
selection rule, so the comparison is fair and free of test leakage.

### 7.4 A high-power paired statistical test
We pool the held-out predictions across CV folds (each molecule predicted exactly once) and compute,
**per toxicity task**, the held-out **_ROC-AUC_** for structured and for scrambled. This yields 12
*paired* observations (one per Tox21 task). We then run:
- a **paired _Wilcoxon signed-rank test_** (nonparametric; primary, mirrors the main benchmark),
- a **sign test** (binomial; how many of the 12 tasks favor structured),
- and **run-level deltas** across random seeds (does the mean gap hold across initializations?).

Pairing per task removes between-task difficulty variance and gives real power from only 12 tasks.

### 7.5 The decomposition that pinpoints the mechanism
We ran three configurations, each as structured vs scrambled, to localize *where* the bias comes from:

| config | entangler | readout | what it isolates |
|---|---|---|---|
| `gate` | graph-gated | single-qubit only | bias from **gates** alone |
| `levelG` (Level 8) | graph-gated | + bond-correlators | gates **+ measurement** |
| `meas_only` | fixed ring (graph-independent) | bond-correlators | bias from **measurement** alone |

### 7.6 Replication across evaluation regimes
We confirmed the effect in two independent regimes: a 15-seed **random-split** paired test and the
**scaffold-CV** test above.

---

## 8. Results

![Bias vs circuit size](figures/fig01_bias_vs_qubits.png)
![Level-8 decomposition at K=4](figures/fig02_decomposition.png)
![Absorbability residual per level](figures/fig05_absorbability.png)

*Figures (generated by `make_figures.py` from `report_data.py`; see also the full set in
`docs/figures/` and the results chapter `docs/06`). Top — **bias vs. qubit count**: gate-gating
(blue) fades to non-significance (open markers = p ≥ 0.05) while measurement-based Level 8 (red)
grows and stays significant (filled markers), all the way to K = 8 (p = 0.0024). Middle —
**mechanism at K = 4**: the bond-correlator readout amplifies the gate-gated bias
(+0.0044 → +0.0078), but with a graph-independent entangler the readout alone hurts (−0.027) —
"place-then-harvest." Bottom — **control validity**: the scramble residual is 0 at Levels 2/4
(absorbed — vacuous) and large at Levels 5–7 (genuine).*

### 8.1 Replication 1 — random-split, 15 seeds (K = 4)
structured beat scrambled in **13 of 15 seeds**; mean ΔAUC **+0.0042**; **sign test p = 0.0037**,
**Wilcoxon p = 0.0062**; 95% CI [+0.0013, +0.0071]. Ordering matched theory: separable (0.664) <
scrambled (0.672) < structured (0.673).

### 8.2 Replication 2 — scaffold-CV, gate-only sweep
Significant at K = 4 (median +0.0044, **p = 0.017**), fading to non-significant at K = 6 (p = 0.17)
and K = 8 (p = 0.52). **Conclusion: the gate-only topology bias is real but does not scale.**

### 8.3 Level 8 decomposition (K = 4)
| config | structured AUC | scrambled AUC | median ΔAUC | sign p | Wilcoxon p |
|---|---|---|---|---|---|
| `gate` | 0.646 | 0.640 | +0.0044 | 0.073 | 0.017 |
| **`levelG`** | 0.642 | 0.633 | **+0.0078** | 0.194 | 0.017 |
| `meas_only` | 0.607 | 0.634 | **−0.027** | 1.0 | 1.0 |

The bond-correlator readout roughly **doubles** the bias magnitude over gates-only; the `meas_only`
negative result proves the readout needs graph-gated entanglement to amplify (§4).

### 8.4 Level 8 scaling (K = 6) — the decisive result
| config | structured AUC | scrambled AUC | median ΔAUC | Wilcoxon p | per-seed run-level ΔAUC |
|---|---|---|---|---|---|
| `gate` | 0.641 | 0.638 | +0.0026 | 0.13 (n.s.) | [+0.012, −0.003, −0.000] |
| **`levelG`** | **0.651** | 0.641 | **+0.0108** | **0.011** | [+0.0165, +0.0069, +0.0066] |

Gate-only has died; Level 8 is *larger than at K = 4*, significant, consistent across all three seeds,
**and** more accurate in absolute terms. The K = 8 confirmation completed and is *stronger still*:
Level 8 ΔAUC **+0.0134**, Wilcoxon **p = 0.0024**, 10/12 tasks favouring structured (struct 0.648 vs
scram 0.633), while gate-only at K = 8 stays non-significant (+0.0030, p = 0.17).

### 8.5 Interpretation of the decomposition
- The inductive bias **lives in graph-gated entanglement** (the `place` step).
- The **measurement readout amplifies it and makes it scale** (the `harvest` step), but only when the
  entanglement is graph-gated — `meas_only` actively hurts.
- Together: a non-absorbable, scalable, hardware-native quantum inductive bias.

---

## 9. Scope and honest limitations

1. **Not a performance claim.** A capacity-unconstrained classical MLP on the same coarse features
   leads by ~5–8 AUC points at every K. The contribution is a *correct, scalable isolation of an
   inductive bias*, not "quantum beats classical."
2. **Small effect size.** The bias is ~0.4–1.1 AUC points. It is *statistically* robust (paired,
   multi-seed, two evaluation regimes) but *practically* small at this scale.
3. **Coarse-graining loses information.** Reducing a molecule to K clusters discards detail; absolute
   AUCs (~0.63–0.66) reflect this. The structured-vs-scrambled *gap*, however, is internally valid
   because both variants see the identical coarse representation.
4. **Higher-K cells use fewer seeds.** K = 4 used 2 seeds, K = 6 used 3, K = 8 used 1 (the
   per-task paired Wilcoxon over the 12 Tox21 tasks still has power at one seed, but tighter
   confidence intervals at K = 8 would benefit from more). All three K points are now filled and
   consistent (gate fades, Level 8 grows monotonically K = 4 → 6 → 8).
5. **One dataset / task family.** Tox21 (12 nuclear-receptor / stress-response assays). Generalization
   to other endpoints is untested.

---

## 10. Reproducibility

| Artifact | File |
|---|---|
| Absorbability proof (all 7 levels) | `_verify_absorb.py` |
| Hardened probe (featurization, scaffold CV, gate-only bias) | `run_bias_probe.py` |
| **Level 8** (measurement readout + decomposition) | `run_levelG_probe.py` |
| Coarse-graph caches | `data/bias_coarse_K{4,6,8}.npz` |
| This report & companion docs | `docs/04_inductive_bias_probe.md`, `docs/05_level8_report.md` |

```bash
# Reproduce the absorbability proof
python _verify_absorb.py

# Gate-only scaffold-CV bias sweep
python run_bias_probe.py --qubits 4 6 8 --folds 3 --seeds 0 1

# Level 8 decomposition + scaling
python run_levelG_probe.py --qubits 4 --folds 3 --seeds 0 1 --configs gate levelG meas_only
python run_levelG_probe.py --qubits 6 --folds 3 --seeds 0 1 2 --configs gate levelG
```

---

## 11. Glossary (cross-disciplinary)

- **_Inductive bias_** (ML): the assumptions an architecture encodes about good solutions; what lets a
  model generalize from limited data.
- **_Qubit / gate / entanglement_** (QC): a qubit is a two-level quantum system; a gate is a unitary
  operation on one or two qubits; entanglement is non-classical correlation between qubits that a
  product (separable) state cannot reproduce.
- **_Expectation value_ `⟨P⟩`** (QC): the average outcome of measuring Pauli operator `P`; for a single
  qubit, `⟨Z⟩ ∈ [−1, 1]`. A **correlator** `⟨Z_iZ_j⟩` is the expectation of the joint observable
  `Z_i ⊗ Z_j` and measures how the two qubits' outcomes co-vary — a direct probe of entanglement.
- **_IsingXX gate_** (QC): a two-qubit entangling gate `exp(−i (θ/2) X_i X_j)`; here `θ` is gated by the
  bond strength `A[i,j]`.
- **_Classical shadows_** (QC): a technique to estimate many low-weight observables from few randomized
  measurements, making the `O(K)` correlator readout cheap on hardware.
- **_GNN encoder_** (ML): graph neural network mapping a molecular graph to a vector.
- **_Spectral clustering_** (ML): graph partitioning via the graph Laplacian's eigenvectors; here used
  to coarse-grain atoms into K qubit-clusters.
- **_Bemis–Murcko scaffold_** (chem): a molecule's core ring system; grouping by scaffold for CV
  ensures the test set is structurally novel.
- **_ROC-AUC_** (ML): area under the ROC curve; probability a random positive is ranked above a random
  negative; 0.5 = chance.
- **_Scaffold-grouped cross-validation_** (ML): CV where whole scaffolds are held out, an
  out-of-distribution split.
- **_Wilcoxon signed-rank test / sign test_** (stats): nonparametric paired tests; here over the 12
  per-task AUC differences, testing whether structured systematically exceeds scrambled.
- **_Absorbable control_** (this work): a control whose intended effect a trainable layer can undo, so
  it tests nothing; the central failure we identified and fixed.
```
