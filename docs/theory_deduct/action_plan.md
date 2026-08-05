# Deductive Action Plan: The Quantum-Topological Information Bottleneck (Q-TIB)

---
**NOTE (2026-07-12):** This document has been superseded by the TC-QIC higher-level theory.
See:
- `docs/theory_deduct/tc_qic_theory.md` — full mathematical theory (TC-QIC framework)
- `docs/theory_deduct/research_program.md` — two-phase research program and action plan
The original 10-phase plan (A-J below) remains as the experimental execution roadmap.
---

This document establishes the theoretical framing and execution plan for analyzing our Quantum Level 8 (Level G) architecture. We transition from benchmarking accuracy to proving a fundamental physical principle: **The Quantum-Topological Information Bottleneck (Q-TIB).**

## The Unified Theory: Q-TIB
Standard Quantum Machine Learning (QML) suffers from barren plateaus because the Hilbert space is too vast to navigate without a prior. Classical Graph Neural Networks (GNNs) suffer from over-smoothing because continuous message passing destroys local variance.

**Q-TIB** solves both by enforcing a double-bottleneck:
1.  **The Topological Bottleneck (Classical Prep):** Coarse-graining compresses the molecule, stripping away high-frequency atomic noise and retaining only the macro-topology (the Adjacency matrix $A$).
2.  **The Operator Geometry Bottleneck (Quantum Harvesting):** The quantum circuit is structurally forbidden from accessing the full Hilbert space. The bond-pooled measurement readout ($\sum A_{ij} \langle Z_i Z_j \rangle$) physically restricts the circuit to *only* harvest correlations that align with the established topology.

This double-bottleneck prevents over-smoothing, guarantees trainability as qubits scale, and produces a structural prior inherently resilient to hardware noise. To rigorously prove the Q-TIB theory, we will execute the following 10-phase pipeline, generating specific datasets and figures to answer scientific questions rather than just producing accuracy tables.

---

## Phase A: Information Preservation Analysis
**Goal:** Prove the Topological Bottleneck compresses data logically, not arbitrarily.
*   **Executable Step 1:** Calculate chemical preservation. Write a script to measure what percentage of aromatic rings, functional groups, and toxicophores survive the Level 8 clustering intact.
*   **Executable Step 2:** Calculate information retention. Compute the Mutual Information and feature variance retained before and after clustering.
*   **Output:** Table summarizing Information Retained vs. Downstream AUC.

## Phase B: Representation Analysis
**Goal:** Understand what the compressed clusters actually represent.
*   **Executable Step 1:** Generate cluster heatmaps and size histograms across the Tox21 dataset.
*   **Executable Step 2:** Output visual samples. Color 10 distinct molecular graphs according to their cluster assignments to visually prove chemical alignment.
*   **Output:** Colored molecular graphs and cluster statistics.

## Phase C: Circuit Analysis
**Goal:** Prove the circuit architecture is structurally sound for the Q-TIB mechanism.
*   **Executable Step 1:** Measure standard circuit metrics: Depth, entangling gate count, and parameter count for $K \in \{4, 6, 8, 10\}$.
*   **Executable Step 2:** Calculate circuit expressibility and effective dimension.
*   **Output:** Plots of Expressibility vs. Performance and Depth vs. Performance.

## Phase D: Learning Dynamics
**Goal:** Track how the Q-TIB constraints affect optimization, proving it prevents Barren Plateaus.
*   **Executable Step 1:** Modify the training loop to log Gradient Norm, Gradient Variance, and Parameter Movement at *every single epoch*.
*   **Executable Step 2:** Compare the loss landscape and convergence rate of Level 8 vs. a generic Hardware Efficient Ansatz (HEA) circuit.
*   **Output:** High-resolution Learning Curves and Gradient Variance Curves.

## Phase E: Quantum Mechanism (The Core Ablations)
**Goal:** Prove that Q-TIB requires *both* structural entanglement placement and structural harvesting.
*   **Executable Step 1:** Entanglement Ablation. Train a "Separable" (No Entanglement) variant.
*   **Executable Step 2:** Measurement Ablation. Train a "Gate-Only" variant with standard single-qubit $\langle Z \rangle$ readouts instead of bond-pooled readouts.
*   **Executable Step 3:** Randomize the observable topology (Scrambled $A$).
*   **Output:** Table comparing Structured, Scrambled, Separable, and Single-Qubit variants.

## Phase F: Generalization & Substrate Analysis
**Goal:** Prove Q-TIB provides a structural regularizer superior to classical models.
*   **Executable Step 1:** Evaluate the Generalization Gap (Train AUC - Test AUC) across strict Scaffold Splits.
*   **Executable Step 2:** Compare Level 8 against a dynamically parameter-matched `ClassicalGNN_pm` to isolate the substrate effect. Analyze performance by specific toxicity categories.
*   **Output:** Performance tables broken down by Scaffold Family and Molecule Size.

## Phase G: Scaling Laws
**Goal:** Understand how Q-TIB behaves as we add qubits.
*   **Executable Step 1:** Collect Information Retained, AUC, Gradient Variance, and Entanglement for $K=4, 6, 8$.
*   **Executable Step 2:** Fit mathematical scaling laws (Linear, Log, Power, Saturation) to the resulting curves.
*   **Output:** Explicit scaling law fits determining if the inductive bias grows or saturates with $K$.

## Phase H: Statistical Causality
**Goal:** Ensure robustness of findings beyond simple $p$-values.
*   **Executable Step 1:** Perform 1,000-resample Bootstrap Confidence Intervals on the resulting AUCs.
*   **Executable Step 2:** Compute effect sizes (Cohen's $d$) for the Structured vs. Scrambled performance gap.
*   **Output:** A rigorous statistical summary table.

## Phase I: Literature Comparison
**Goal:** Frame our findings as a scientific dialogue with prior work.
*   **Executable Step 1:** Create a definitive table comparing Q-TIB findings against QML literature expectations.
    *   *Example:* Lit expects deeper circuits are better $\rightarrow$ We show trainability limits dominate.
    *   *Example:* Lit expects entanglement is always beneficial $\rightarrow$ We show placement/alignment matters more.
*   **Output:** A "Literature vs. Evidence" Markdown table.

## Phase J: Theory Extraction (The Final Synthesis)
**Goal:** Consolidate Phases A-I into Chapter 5 of the thesis.
*   **Execution Rule:** Only after all 9 phases are complete will we synthesize the theory. The structure will be:
    1.  *Observation* (Empirical pattern)
    2.  *Evidence* (Converging analyses from A-H)
    3.  *Mechanistic Explanation* (Q-TIB)
    4.  *Boundary Conditions* (When does it fail?)
    5.  *Relation to Prior Work* (Phase I).

---

## Phase K: Quantum Model Refinements -- Architecture Evolution

*Derived from code analysis of `run_levelG_probe.py`, `_alt_b_probe.py`, and `src/quantum_levels.py`.
Each route is classified by: implementation cost (LOW/MED/HIGH), expected bias impact (LOW/MED/HIGH),
and theory connection.*

### K1: Extended Bond-Pooled Observable Set
**Cost:** LOW | **Impact:** MED | **Theory:** TC-QIC Master Theorem (observable completeness)
- **Current:** bond-pool only ZZ and XX correlators
- **Route:** add YY, XZ, YZ bond-pooled correlators to the readout feature vector
- **Rationale:** ZZ ~ charge-charge, XX ~ resonance/flip-flop, YY ~ delocalization (aromatic rings); each Pauli product probes a chemically distinct interaction channel
- **Implementation:** in `GraphG.forward`, extend `obs` to include `qml.expval(qml.PauliY(i) @ qml.PauliY(j))` for all pairs and bond-pool them
- **Pass/fail:** if structured > scrambled persists for YY pooling on aromatic-rich subsets, confirm channel-specificity

### K2: Multi-Hop Bond Pooling
**Cost:** LOW | **Impact:** MED | **Theory:** Scaling lemma (k-hop graph signal capture)
- **Current:** single-hop `b[i] = sum_j A[i,j] * corr(i,j)` (direct bonds only)
- **Route:** 2-hop pooling via `A2 = A @ A` (normalized): `b2[i] = sum_j A2[i,j] * corr(i,j)` concatenated with b1
- **Rationale:** conjugation paths and through-space effects act over 2+ bonds; second-neighbor pooling is the quantum correlator analog of 2-hop GNN aggregation
- **Implementation:** precompute `A2 = torch.bmm(adj, adj)` in `_bond_pool`, add as extra feature block in head input
- **Pass/fail:** K=8 structured-scrambled gap should widen for multi-hop vs single-hop on tasks with extended conjugation (NR-AhR, SR-ARE)

### K3: Bond-Order Conditional Gate Selection
**Cost:** MED | **Impact:** HIGH | **Theory:** TC-QIC chemical topology prior
- **Current:** single gate family IsingXX for all bond types
- **Route:** encode bond order explicitly: single bond -> IsingZZ, double bond -> IsingXX, aromatic (order ~1.5) -> IsingYY; weighted mixture via bond order `o`: `(1-o)*IsingZZ + o*IsingXX`
- **Rationale:** directly extends the Level 2 motif->operator correspondence (RY/RZ/IsingXX) to the Level G graph-native setting; the gate Hamiltonian aligns with the physical bond Hamiltonian type
- **Implementation:** featurize coarse adjacency with two matrices: `A_order[i,j]` (bond order), `A_arom[i,j]` (aromaticity fraction per cluster pair); condition gate type selection on these
- **Pass/fail:** tasks requiring aromatic ring discrimination (NR-ER, NR-AhR) should show larger gain

### K4: Degree-Normalized Bond Pooling
**Cost:** LOW | **Impact:** LOW-MED | **Theory:** permutation equivariance of readout
- **Current:** sum aggregation `b[i] = sum_j A[i,j] * corr(i,j)` -- hub nodes (high degree) dominate
- **Route:** mean aggregation `b[i] = (sum_j A[i,j] * corr(i,j)) / (sum_j A[i,j] + eps)` -- normalizes by weighted degree
- **Rationale:** molecules vary in size K; mean pooling makes the feature molecule-size-invariant, reducing variance for small K where one qubit may aggregate many bonds
- **Implementation:** one-line change in `GraphG._bond_pool`; compare structured-scrambled gap with and without normalization

### K5: Graph-Conditional Data Re-Uploading
**Cost:** MED | **Impact:** MED-HIGH | **Theory:** Q-IB expressibility bound
- **Current:** encoding `enc[0]*ry, enc[1]*rz` uses same scale every layer regardless of adjacency
- **Route:** per-layer adjacency-aggregated shift: `ry_l[i] = ry[i] + gamma_l * (sum_j A[i,j] * rz[j])` before the RY gate -- qubit i's encoding is conditioned on its graph neighborhood at each layer
- **Rationale:** makes the encoding itself topology-aware (not just the entanglement), creating a deeper bias injection point; analogous to positional encoding in graph transformers
- **Implementation:** add per-layer `gamma` parameter (n_layers scalar), compute neighbor-aggregated shift before each encoding block in `GraphG`'s qnode
- **Pass/fail:** gradient variance at initialization should increase (richer data dependence), barren plateau risk should decrease due to structure guidance

### K6: Learnable Measurement Basis
**Cost:** HIGH | **Impact:** HIGH | **Theory:** TC-QIC measurement bottleneck (Section 1)
- **Current:** measure in fixed XYZ Pauli basis
- **Route:** apply per-qubit learnable SU(2) rotation `U(phi, theta, psi)` before measurement; optimize measurement basis jointly with circuit
- **Rationale:** the optimal measurement basis for topology-aligned quantum correlations may not be the Pauli eigenbasis; allowing the basis to adapt is the quantum analog of learning the kernel in kernel methods
- **Implementation:** add `meas_rot` parameters (n_qubits, 3); apply `qml.Rot(meas_rot[i])` on each qubit before `qml.expval`; use a non-absorbable control (scramble the measurement basis assignment, not the adjacency)
- **Constraint:** keep the bond-pooling AFTER the rotated measurement to preserve non-absorbability

### K7: Quantum Natural Gradient for Circuit Parameters
**Cost:** MED | **Impact:** MED | **Theory:** TC-QIC barren plateau resistance (Section 4)
- **Current:** AdamW for all parameters; circuit params (theta, ringp, pairp) treated classically
- **Route:** use SPSA or block-diagonal QNG (via PennyLane's `qml.QNGOptimizer`) for the quantum circuit params; keep AdamW for classical head and encoder
- **Rationale:** QNG follows the quantum Fisher information metric, which is the natural geometry for parametric quantum circuits; this directly tests the TC-QIC barren-plateau-resistance claim via the structured Riemannian gradient
- **Implementation:** split optimizer: `qml.QNGOptimizer(stepsize=0.01)` for `[theta, ringp, pairp, enc]`, `torch.optim.AdamW` for `[feat, head]`; requires PennyLane >= 0.36
- **Pass/fail:** gradient variance over training should be higher than AdamW baseline (less BP suppression)

### K8: Symmetry-Tied Variational Block
**Cost:** LOW | **Impact:** LOW-MED | **Theory:** generalization bound (hypothesis class restriction)
- **Current:** `theta[l, i, *]` is fully independent per qubit -- breaks all molecular symmetry
- **Route:** for qubits in the same cluster-type equivalence class (e.g., all degree-2 aliphatic carbon clusters), tie the theta initialization (soft prior, not hard constraint)
- **Rationale:** reduces effective parameter count by molecular symmetry class; tighter hypothesis class -> better generalization bound; most directly tests the "equivariance to molecular symmetry" corollary
- **Implementation:** after spectral clustering, compute cluster type (dominant atom type + degree); group qubits by type; shared initialization with small noise; optionally add weight-sharing penalty term

### K9: Multi-Scale Coarse-Graining Ensemble
**Cost:** HIGH | **Impact:** HIGH | **Theory:** TC-QIC phase diagram (Section 4)
- **Current:** single coarse-graining scale K=n_qubits
- **Route:** run K=4, 6, 8 circuits in parallel (or sequentially), bond-pool at each scale, concatenate features before the classification head
- **Rationale:** the TC-QIC phase diagram predicts a capacity-topology alignment sweet spot; an ensemble over K combines the strong bias of small K with the richer correlator space of large K without picking one
- **Implementation:** three `GraphG` instances with shared `feat` encoder, independent circuit params; concatenate bond-pooled outputs into a unified head
- **Pass/fail:** multi-scale AUC should exceed best single-K by >0.005; structured-scrambled gap should be larger than any single K

### K10: Classical Equivariant Baseline Strengthening
**Cost:** MED | **Impact:** HIGH | **Theory:** Phase 2 experimental validation
- **Current:** `ClassicalGNN` uses simple `h_i * (sum_j A_ij h_j)` product interaction
- **Route:** upgrade to a proper equivariant GNN: 2-layer E(n)-equivariant graph convolution (e.g. EGNN-style) with bond-type edge features, parameter-matched to quantum Level G at each K
- **Rationale:** current classical baseline is intentionally weak (structural analog only); a stronger equivariant classical model tests the TC-QIC claim that quantum correlators provide signal BEYOND classical graph message-passing
- **Implementation:** implement `ClassicalEquivGNN` in `run_levelG_probe.py`; use EGNN update rule `h_i' = phi(h_i, sum_j m_ij)` with `m_ij = psi(h_i, h_j, A_ij, ||h_i - h_j||^2)`; add to `CONFIGS` dict

---

## Refinement Priority Order

For maximum theory-experiment coherence, execute in this order:

| Priority | Route | Reason |
|----------|-------|--------|
| 1 | K4 (degree-norm pooling) | one-line fix, removes size confound from all further experiments |
| 2 | K1 (extended observables) | directly tests observable completeness; low cost, high theory payoff |
| 3 | K3 (bond-order conditional gates) | extends Level 2 operator-family logic to Level G; tests TC-QIC chemical topology prior |
| 4 | K2 (multi-hop pooling) | tests scaling lemma prediction that 2nd-neighbor structure adds signal |
| 5 | K10 (strong classical baseline) | required to make any quantum-vs-classical claim publishable |
| 6 | K7 (QNG optimizer) | tests barren-plateau resistance claim from TC-QIC Section 4 |
| 7 | K5 (graph-conditional re-upload) | most novel; requires K1/K3 results first to interpret correctly |
| 8 | K6 (learnable measurement basis) | theoretically rich but high implementation cost; defer to post-submission |
| 9 | K8 (symmetry-tied init) | low impact but directly tests generalization bound corollary |
| 10 | K9 (multi-scale ensemble) | high impact but expensive; use as final ablation if K=4/6/8 results are clean |
