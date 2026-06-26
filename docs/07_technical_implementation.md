# Technical Implementation & Reproducibility

*Companion to [`06_results_benchmarking.md`](06_results_benchmarking.md). This document records the
**mechanisms** behind the results — circuit construction, data re-uploading, the graph-gated
entangler, the measurement readout, the control variants, the training regime, and the evaluation
protocol — at a level of detail sufficient to reproduce or re-implement the experiments. Code
references are `file:line` into this repository.*

> **Which stack produced the headline results?** The statistically valid results in the report come
> from the **probe / Level-8 stack** (`run_bias_probe.py`, `run_levelG_probe.py`) — a deliberately
> minimal *qubits-as-graph-nodes* design. The seven abstract-qubit **benchmark levels**
> (`src/quantum_levels.py`, `run_benchmark.py`) are documented here too because they share most
> techniques (data re-uploading, the variational block, the X/Y/Z readout, the scramble control) and
> because their *absorbability* is the methodological finding that motivated the probe. Where a
> technique differs between the two stacks, both are given.

---

## 1. Featurization

### 1.1 Probe / Level-8: molecule → coarse quantum graph
Source: `run_bias_probe.py:42` (`coarse_graph`), `:87` (`random_adj_like`), `:266` (`standardize`).

A molecule (SMILES) is reduced to **K clusters** (K = #qubits), one cluster per qubit:

1. Parse with RDKit; compute Gasteiger partial charges (`AllChem.ComputeGasteigerCharges`, non-finite
   → 0). Per-atom feature vector (`FDIM = 5`):
   `[atomic_number, gasteiger_charge, degree, is_aromatic, in_ring]`.
2. Build the atom-level weighted bond graph `A_atom[i,j] = bond_order`.
3. **Spectral clustering** of the atoms into K clusters
   (`SpectralClustering(n_clusters=K, affinity='precomputed', assign_labels='discretize',
   random_state=0)` on `A_atom + 1e-6`); if `#atoms ≤ K`, clusters are assigned round-robin.
4. **Per-qubit features** `QF ∈ ℝ^{K×5}` = the mean of each cluster's atom features.
5. **Coarse adjacency** `A ∈ ℝ^{K×K}`: `A[c,c'] = Σ bond_orders between clusters c,c'`, then
   normalised by its max (`A /= A.max()+1e-9`). This is the molecule's topology, **as fixed data**.

The cached tensors per K live in `data/bias_coarse_K{4,6,8}.npz` with keys
`QF (N,K,5), AT (N,K,K) true adj, AR (N,K,K) random adj, Y (N,12), SCAF (N,)`.

**Feature standardisation** is done *inside CV*, fit on the training fold only
(`standardize`, `run_bias_probe.py:266`): `QF ← (QF − μ_tr)/(σ_tr + 1e-6)` over the flattened
`(·,5)` features — no leakage from val/test.

### 1.2 The scrambled adjacency (the control's data)
Source: `run_bias_probe.py:87` (`random_adj_like`).

`AR` keeps **exactly** the multiset of off-diagonal bond weights of `AT` but shuffles which pair holds
which weight (`rng.shuffle` over the upper triangle, symmetrised). So edge **count** and weight
**multiset** (hence density) are preserved per molecule; only the **topology** changes. The seed is
the molecule index `i`, so `AR` is deterministic and fixed per molecule across epochs/folds.

### 1.3 Benchmark (7-level) featurization
Source: `src/data_loader.py` (`mol_to_pyg`, `build_merged_dataframe`), `src/features/semantic_extractor.py`.

Full molecular graphs: 5 categorical atom features `[Z, degree, formal charge, total-H, aromatic]`
(vocab `(120,10,7,5,2)`), 8 continuous `x_cont`
`[partial_charge, electronegativity, x, y, z, is_donor, is_acceptor, is_hydrophobe]` (RDKit ETKDG 3D
conformer + Gasteiger + pharmacophore tags), 3 categorical bond features + 1 continuous 3D distance.
Cached to `data/featurized_Tox21_ToxCast.pt`. The **merged** dataset keys molecules by canonical
SMILES; a molecule absent from one source gets `NaN` labels there (ignored by the masked loss).

---

## 2. The classical encoder (benchmark stack only)
Source: `src/features/semantic_extractor.py` (`SemanticFeatureExtractor`).

Upstream of every *benchmark* quantum level (not used by the probe). Pipeline:
* Embed 5 categorical atom features (`emb_dims=(64,16,8,8,4)`), project to `hidden_dim=64`, dropout.
* **Bond-aware message passing:** edge embedding = 3 categorical bond embeddings + continuous
  distance → `Linear → hidden_dim`; **3 × GINEConv** (`train_eps=True`) with `LayerNorm` residuals:
  `h ← norm(h + relu(conv(h, edge_index, edge_emb)))`.
* **Four attentional-pooling heads** (`AttentionalAggregation`) → four molecule vectors (B,64):
  **motif** (local), **cycle** (+aromatic embedding), **spectral** (+degree embedding), **chem**
  (+8 continuous `x_cont`).
* An auxiliary **descriptor head** (`Linear(256→64→6)`) predicts 6 molecular descriptors; its MSE is
  added to the loss so gradients refine the shared embeddings.

The probe stack deliberately replaces this whole encoder with a **fixed, tiny** `Linear(5→2)`
(`run_bias_probe.py:164`) so that no expressive trainable layer can re-route topology and re-absorb
the control (see §6).

---

## 3. Quantum circuit construction — the core techniques

These primitives are shared (the benchmark levels in `src/quantum_levels.py:40-64`; the probe in
`run_bias_probe.py:139` / `run_levelG_probe.py:37`).

### 3.1 Data re-uploading
Source: `src/quantum_levels.py:43` (rationale), per-level `for l in range(n_layers)` encoding blocks;
probe `run_bias_probe.py:149`.

The chemistry→operator **encoding is re-applied at every one of the `n_layers` (=2) layers**, not
just at the input. The original design encoded data once into the variational weights and measured
only ⟨Z⟩, which "collapsed molecular variation and made phase-based biases unmeasurable"
(`quantum_levels.py:41-45`). Re-uploading interleaves data-dependent encoding with a *separate*
trainable variational block, increasing the circuit's expressivity in the data and exposing
phase-encoded signal. In the probe, each layer applies `RY(enc₀·ry_i), RZ(enc₁·rz_i)` from the
(fixed-encoded) features, then the entangler, then the variational `RY/RZ` + ring.

### 3.2 Learnable encoding scales (`enc_scale`)
Source: e.g. `quantum_levels.py:258, 407, 523`; probe `self.enc = nn.Parameter(torch.ones(2))`
(`run_bias_probe.py:168`).

Each modality's encoding angle is multiplied by a learnable scalar (init `1.0`). This lets training
**down-weight a modality whose operator assignment is wrong**, which makes the structured-vs-scrambled
test *conservative* (the scrambled model can mute a misrouted feature instead of being forced to use
it). It does **not**, by itself, rescue an absorbable control (§6).

### 3.3 The trainable variational block
Source: `quantum_levels.py:49` (`_variational_block`); probe inline.

After each encoding block: `RY(θ_{l,i,0}), RZ(θ_{l,i,1})` per qubit, then — if entanglement is enabled
— a **CRZ ring** `CRZ(ent_{l,i}, [i, (i+1)%K])`. `θ` and `ent` are the data-independent trainable
parameters. Init scale is small (`randn·0.1`, `run_bias_probe.py:165-167`) so the circuit starts near
identity and gradients are well-conditioned.

### 3.4 Richer X/Y/Z readout
Source: `quantum_levels.py:60` (`_xyz_measure`); probe `run_bias_probe.py:160-162`.

Every qubit is measured in **three** bases ⟨X⟩,⟨Y⟩,⟨Z⟩ (→ `3K` features), not ⟨Z⟩ only. Circuits whose
encoding lives in phase (`RZ`, `IsingXX`) produce ≈0 ⟨Z⟩ at init and therefore no gradient; ⟨X⟩,⟨Y⟩
expose that phase information. This was a prerequisite for any bias to be measurable at all.

### 3.5 Graph-gated entanglement (probe / Level 8 — the non-absorbable injection)
Source: `run_bias_probe.py:153`, `run_levelG_probe.py:51`.

```python
for (i, j) in PAIRS:                       # all qubit pairs
    qml.IsingXX(adj[:, i, j] * pairp[l, pidx], wires=[i, j])
```
The entangling angle is the **per-molecule adjacency entry `A[i,j]` times a trainable scalar
`pairp`**. A trainable scalar can scale a coupling but cannot create a bond where `A[i,j]=0`, so the
entanglement pattern *is* the molecular graph and is **data, not a weight** — the property that makes
the scramble non-absorbable (§6). `adj` is broadcast over the batch dimension (§3.7).

### 3.6 Measurement-based bond-correlator readout (Level 8 — the novelty)
Source: `run_levelG_probe.py:59-65` (observables), `:75-82` (`_bond_pool`), `:84-94` (forward).

In addition to the `3K` single-qubit expectations, Level 8 measures **two-qubit correlators**
⟨Z_iZ_j⟩, ⟨X_iX_j⟩ for every pair, then **bond-pools** them along the real graph:
```python
b[i] = Σ_j A[i,j] · ⟨corr(i,j)⟩          # implemented via index_add over pair lists
```
The per-molecule feature vector is `[⟨X_i⟩,⟨Y_i⟩,⟨Z_i⟩, b_ZZ[i], b_XX[i]]` → `5K` numbers → one
trainable `Linear(5K → 12)` head. Because `A` multiplies the correlator **before** the only trainable
layer and varies per molecule, no weight reparametrisation can re-select which physical correlator was
summed in — the readout is non-absorbable and is an O(K) (sparse-graph), ≤2-local, hardware-native
graph aggregation. The bond pooling is symmetric (`index_add` on both `i` and `j`) and
permutation-invariant.

### 3.7 Batched circuits via parameter broadcasting
Source: probe circuits pass `ry[:, i]`, `adj[:, i, j]` (leading batch dim); benchmark notes in
`README.md` ("batched via parameter broadcasting (~50× faster)").

Inputs carry a leading batch axis and PennyLane broadcasts the QNode over it, so a whole minibatch is
evaluated in one circuit call instead of per-molecule — the single change that made CV over ~7.8k
molecules tractable on CPU. Device: `default.qubit`.

---

## 4. The control variants — exact definitions

| Variant | Probe code | Benchmark code | What it changes vs structured |
|---------|-----------|----------------|-------------------------------|
| **structured** | `adj = AT` (`run_levelG_probe.py:100`) | `ansatz='strong'`, identity perms | the true topology / true chemistry→qubit map |
| **scrambled** | `adj = AR` (random adj) | `ansatz='scrambled'` → `_perms(...)` | topology shuffled (probe) / per-site fixed random permutations (benchmark) |
| **separable** | `GraphQ(variant='separable')` → `entangle=False` (`run_bias_probe.py:143`) | `ansatz='separable'` → `entangle=False` (`quantum_levels.py:235`) | **all** entangling gates removed (no IsingXX, no CRZ ring) |
| **meas_only** | `CONFIGS['meas_only']` → `entangler='fixed'` (`run_levelG_probe.py:53`) | — | graph-gated entangler replaced by a fixed graph-independent **ring** IsingXX |
| **classical** | `ClassicalRef` MLP (`run_bias_probe.py:177`) | `Level{N}Classical`, inner-dim param-matched (`run_benchmark.py:39`) | classical baseline |

Notes:
* **Benchmark scramble** (`_perms`, `quantum_levels.py:10`): per gate-site a *different* fixed random
  permutation, never the identity, consecutive sites forced to differ — preserves every gate, weight,
  depth and the entanglement pattern; only the "feature i ↔ qubit i" alignment is destroyed.
* **Probe classical** (`ClassicalRef`, `:177`) is a 2-hidden-layer MLP (`h=32`) on
  `[flatten(QF) ‖ flatten(A)]`; it is **capacity-unconstrained** (≈9–13× the quantum parameter count)
  and is a *context* baseline, not a parameter-matched one.
* **Benchmark classical** *is* parameter-matched: `match_classical_inner_dim`
  (`run_benchmark.py:39`) solves the MLP inner width so its parameter count equals the quantum model's
  (`@lru_cache`d across folds).

---

## 5. Training regime

### 5.1 Probe / Level-8 (produced the headline numbers)
Source: `run_bias_probe.py:221` (`train_eval`), `run_levelG_probe.py:97`.

* **Loss:** masked BCE only (`masked_bce`, `:188`) — `binary_cross_entropy_with_logits` with per-task
  `pos_weight`, masked over `NaN` labels and normalised by the valid-entry count.
* **`pos_weight`** (`:203`) `= n_neg/n_pos` per task, **capped at 20**, computed on the **training**
  fold only.
* **Optimiser:** `AdamW`, `weight_decay=1e-4`, **two parameter groups** — circuit params
  (`theta, ringp, pairp, enc`) at **lr 1e-2**, everything else (the `Linear` head/encoder) at
  **lr 1e-3**. The small VQC trains far slower than the linear parts at a shared LR, so it needs the
  higher rate (`src/train.py:10-13`).
* **Epoch selection without test peeking** (`:234-248`): after each epoch, score the
  **scaffold-disjoint validation** set; whenever validation ROC improves, snapshot the **test-fold
  probabilities at that epoch**. The returned test prediction is the one at the best-validation epoch
  — the test fold is never used for selection. `epochs=18–30`, `batch=128`.
* **Feature squashing:** `a = atan(Linear(QF))` before the circuit (`run_bias_probe.py:172`) keeps
  encoding angles in `(−π/2, π/2)`.

### 5.2 Benchmark (7-level) trainer
Source: `src/train.py:9` (`Trainer`), `run_benchmark.py:304-331`.

* **Composite loss:** `BCE + 0.1·SupervisedContrastive + 0.1·DescriptorMSE`
  (`MaskedBCEWithLogitsLoss` + `MultiTaskSupervisedContrastiveLoss(temperature=0.07)` + `MSELoss`).
* Same **dual-LR AdamW** (q-keys `q_layer,q_motif,q_cycle,q_spectral` at 1e-2; rest 1e-3; wd 1e-4),
  **gradient clipping** `max_norm=1.0`, `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)` on
  validation loss.
* **Early stopping** on the **primary-block (Tox21) validation ROC** (a classification signal, not
  the mixed loss), `patience=15`; best weights restored before test (`run_benchmark.py:313-331`).
* **Same init seed across model types within a fold** (`torch.manual_seed(1000*fold+7)`,
  `run_benchmark.py:306`) so variants are comparable.

### 5.3 Hyperparameter summary

| Item | Probe / Level 8 | Benchmark 7-level |
|------|-----------------|-------------------|
| qubits K | 4, 6, 8 | 4, 6 |
| circuit layers | 2 | 2 |
| encoding | fixed `Linear(5→2)`, `atan` squash | trainable GNN + `Linear(64→K)` |
| readout | ⟨X,Y,Z⟩ (+ ⟨ZZ⟩,⟨XX⟩ bond-pooled for Level 8) | ⟨X,Y,Z⟩ |
| loss | masked BCE | BCE + 0.1·contrastive + 0.1·desc-MSE |
| optimiser | AdamW, wd 1e-4 | AdamW, wd 1e-4, grad-clip 1.0 |
| LR (circuit / rest) | 1e-2 / 1e-3 | 1e-2 / 1e-3 |
| `pos_weight` cap | 20 | 20 |
| epochs / selection | 18–30, best-val-ROC snapshot | ≤100, early-stop patience 15 |
| batch size | 128 | 128 |
| device / simulator | CPU / `default.qubit` | CPU / `default.qubit` |

---

## 6. Why the controls are (or are not) valid — the absorbability mechanism
Source: `_verify_absorb.py`; report §I.2; `docs/04`, `docs/05` §2.

A scramble that permutes the **output of a free trainable layer** is undone by training:
`P·(W·x) = (P·W)·x = W′·x`, so the optimiser learns `W′` and reaches the structured optimum exactly.
`_verify_absorb.py` tests this with **no training**: it builds the structured and scrambled circuits
with **identical variational weights**, feeds the scrambled circuit the correspondingly **permuted
input**, and measures `residual = max|structured − scrambled(permuted)|`. `residual = 0` ⇒ a single
input permutation reproduces structured exactly ⇒ a free upstream projection realises it for free ⇒
the control is **vacuous**. Result: bit-exact 0 at benchmark Levels 1/2/4 (each input passes through
one permutation behind its own free `Linear`), nonzero only where one vector is reused under several
inconsistent permutations (Levels 5–7).

The probe/Level-8 design evades this two ways (`docs/04` §5): (a) **reuse** — feed one vector through
conflicting gates (what Levels 5–7 do), and (b) **fixed data** — inject the adjacency `A` as
per-molecule data upstream of the only trainable layer (Level 8). Only (b) also yields a *scalable
graph* bias.

---

## 7. Evaluation protocol

* **Scaffold-grouped CV** (`scaffold_folds`, `run_bias_probe.py:251`; benchmark
  `run_benchmark.py:219`): Bemis–Murcko scaffold per molecule (`MurckoScaffold`), `GroupKFold` so
  whole scaffolds are held out (OOD). **Validation** is carved from the *training* scaffolds and kept
  scaffold-disjoint (`val_frac≈0.10–0.15`) so early stopping is not tuned on leaked structures.
* **Pooled per-task scoring:** every molecule is predicted exactly once per seed-CV; per-task ROC-AUC
  is computed on the pooled predictions, skipping single-class tasks (`per_task_auc`, `:211`). Per-seed
  per-task AUCs are averaged to 12 stable paired observations.
* **Primary test:** per-task **paired Wilcoxon signed-rank**, one-sided (`structured > scrambled`),
  over the 12 Tox21 tasks — pairing cancels between-task difficulty and gives power from only 12
  tasks. Reported with a **sign test** (binomial over the 12 task signs) and **run-level ΔAUC** across
  seeds.
* **Second regime:** a 15-seed **random-split** paired test (`_alt_b_multiseed.py`) replicates the
  K=4 effect (mean ΔAUC +0.0042, Wilcoxon p=0.0062, 95% CI [+0.0013,+0.0071]).
* **Benchmark extras:** bootstrap 95% CIs (`bootstrap_metrics`, `run_benchmark.py:136`, 500 resamples,
  20 random tasks each), AUPRC and Brier alongside ROC-AUC, Bonferroni ×3 over the three comparisons,
  per-source-block reporting (Tox21 vs ToxCast).

---

## 8. Reproducibility

### 8.1 Environment (verified)
```
python 3.12.10 · torch 2.12.0+cpu · torch_geometric 2.8.0 · pennylane 0.45.0
rdkit 2026.03.3 · numpy 2.4.6 · scipy 1.17.1 · scikit-learn 1.9.0 · pandas 3.0.3
```
All experiments run on CPU (`default.qubit` statevector). No GPU/quantum-hardware dependency.

### 8.2 Determinism
Per-fold/seed `torch.manual_seed`; `random_adj_like` seeded by molecule index; spectral clustering
`random_state=0`; `_perms` seeded (`20240617`). The coarse-graph caches make featurisation
deterministic and fast on re-run. Residual nondeterminism is BLAS-level float reduction order only.

### 8.3 Commands
```bash
# Absorbability proof (no training; bit-exact)
python _verify_absorb.py

# Gate-only topology bias vs qubits (Model A)
python run_bias_probe.py   --qubits 4 6 8 --folds 3 --seeds 0 1

# Level-8 decomposition + scaling (Models B, C and the gate row)
python run_levelG_probe.py --qubits 4 --folds 3 --seeds 0 1   --configs gate levelG meas_only
python run_levelG_probe.py --qubits 6 --folds 3 --seeds 0 1 2 --configs gate levelG

# Report figures (deterministic, from report_data.py)
python make_figures.py

# Optional reduced-fidelity corroboration npz (CPU-heavy; run alone)
python make_figdata.py --folds 3 --seeds 0 1 --epochs 18
```

### 8.4 File map

| Artifact | File |
|----------|------|
| Coarse-graph featuriser + probe circuit (Model A / D / classical) | `run_bias_probe.py` |
| Level-8 circuit, bond-correlator readout, decomposition (B / C) | `run_levelG_probe.py` |
| 7-level circuits, re-uploading, `_perms`, variational block | `src/quantum_levels.py` |
| GNN encoder (benchmark stack) | `src/features/semantic_extractor.py` |
| Trainer (dual-LR, composite loss, early stop) | `src/train.py` |
| Featurisation, scaffold, merge | `src/data_loader.py` |
| Absorbability proof | `_verify_absorb.py` |
| Single source of all reported numbers | `report_data.py` |
| Figure generation | `make_figures.py` → `docs/figures/` |
| Reduced-fidelity reproduction harness | `make_figdata.py` → `results/figdata/` |
| Coarse-graph caches | `data/bias_coarse_K{4,6,8}.npz` |
| Raw run logs (headline numbers) | `_levelG_k{4,6,8}.log`, `_sweep.log` |

---

*Every quantitative claim in `docs/06` is reproducible from the commands in §8.3 and traces to
`report_data.py`, which names the source log for each value.*
