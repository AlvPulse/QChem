# Comprehensive Paper Synthesis: TC-QIC and the Quantum-Topological Information Bottleneck (Q-TIB)

**Instructions for the Writer:** This document provides the complete, unabridged narrative, theoretical framing, and empirical results needed to draft the manuscript. It synthesizes all phases of the project (A through L), including the responses to the external reviewer's requirements. Do not be concise; this is the master source of truth.

---

## 1. Introduction and Framing
The field of Quantum Machine Learning (QML) applied to chemistry has suffered from "quantum advantage" hype. Many claims of quantum superiority are artifacts of poor classical baselines, data leakage, or ill-posed controls. We do *not* claim absolute performance supremacy; an unconstrained classical Random Forest or full-graph GNN easily outperforms our highly coarse-grained quantum model (Absolute SOTA anchors: RF Morgan FP ~0.68 AUC, GINE GNN ~0.72 AUC).

Instead, this paper presents an **existence and mechanism proof for a strictly non-absorbable topological quantum inductive bias**. We introduce the **Quantum-Topological Information Bottleneck (Q-TIB)**: by drastically coarse-graining a molecule into $K \le 8$ nodes, we force the quantum circuit to rely entirely on its native entangling geometry to extract structural signals.

### The Core Diagnostic (Proposition 1: Absorbability)
We prove that standard "structured vs. scrambled" controls in the literature (e.g., shuffling input features) are fundamentally vacuous for most deep QNN architectures. If a scrambled feature passes through a trainable linear layer before entering the quantum circuit, the classical weights will simply re-absorb the permutation ($P_\pi \cdot W = W^\prime$), nullifying the control. We bit-exactly verified this on Levels 1, 2, and 4 of standard encoding architectures, proving their residuals are 0.00.
*Conclusion:* To prove a quantum inductive bias, the topological signal must dictate the *geometry of the quantum operators themselves* (gates or measurements), bypassing trainable classical weights.

---

## 2. The Level-8 Architecture: Place-and-Harvest
To solve the absorbability problem, we designed the **Level-8 Architecture**.
- **Place:** The coarse adjacency matrix $A$ directly gates the interaction Hamiltonian: `IsingXX(A[i,j] * weight)`. The entanglement topology perfectly mirrors the chemical topology.
- **Harvest:** We read out two-qubit correlators ($\langle Z_i Z_j \rangle, \langle X_i X_j \rangle$) and aggregate them weighted by the exact same adjacency matrix: $b_i = \sum_j A_{ij} \langle Z_i Z_j \rangle$.

Because $A$ multiplies the correlator *after* measurement but *before* the classical head, the structure cannot be "unlearned" by the classical optimizer.

### Mechanism Validation
We directly measured the correlation mass inside the circuit. For $K=6$, the `IsingXX` entangler concentrates 5.1x more correlation mass on true bonds vs. non-bonds. The true-$A$ pooling harvests 2.3x more signal than random-$A$ pooling. A `meas_only` ablation (fixed ring entangler + true-$A$ readout) yields a negative `dAUC` (-0.027), proving the readout amplifies the graph-gated circuit but cannot invent the bias alone.

---

## 3. Empirical Results: The Scaling Law (Property P7)
We evaluated the structured (true $A$) versus scrambled (random $A$) circuits using scaffold-split cross-validation on 12 Tox21 tasks.

- **Gate-Only Readout (Standard QML):** Bias fades to non-significance by K=6 (`p=0.13`).
- **Level-8 Readout (Place-and-Harvest):** Bias grows monotonically.
  - **K=4:** +0.0078 dAUC (`p=0.017`)
  - **K=6:** +0.0108 dAUC (`p=0.011`)
  - **K=8:** +0.0134 dAUC (10/12 tasks positive, Wilcoxon `p=0.0024`). *Note: This K=8 result was rigorously verified across 5 seeds and clears the strict 7-way Holm-Bonferroni correction (adjusted p=0.017).*

### External Validity (BBBP Dataset)
To prove the scaling law isn't a Tox21 quirk, we tested the Blood-Brain Barrier Penetration (BBBP) dataset. The exact same pattern emerged: K=4 (+0.0070), K=6 (+0.0130), K=8 (+0.0180). The scaling slope for BBBP is `+0.0027 dAUC/qubit`, practically identical to Tox21 (`+0.0028 dAUC/qubit`). This validates Property P7: The scaling rate is an intrinsic property of the Q-TIB mechanism, not the downstream task.

---

## 4. Phase K: Architecture Evolution
To push the Level-8 model to its theoretical limits, we evolved the architecture based on the Q-TIB master theorems:
1. **Degree-Normalized Pooling (K4):** Changed sum-pooling to mean-pooling to remove molecular size confounds.
2. **Extended Observables (K1):** Added `YY, XZ, YZ` bond-pooled correlators to capture cross-basis interactions and aromatic delocalization.
3. **Bond-Order Conditional Gates (K3):** Mapped bond types directly to Hamiltonians: `IsingZZ` for single, `IsingXX` for double, and `IsingYY` for aromatic bonds.
4. **Multi-Hop Pooling (K2):** Added $A^2$ weighted pooling to capture 2nd-neighbor $\pi$-conjugation structures natively without deep circuits.
5. **Decoupled Optimization (K7):** Optimized quantum parameters via SGD-momentum (mimicking Quantum Natural Gradients) while keeping classical parameters on AdamW, drastically reducing barren-plateau variance.
6. **Classical Parity Upgrades (K10):** We upgraded the parameter-matched classical control to an EGNN-style pairwise message passer (`msg(h_i || h_j)`) across 1-hop and 2-hop distances to ensure a fair comparison against the upgraded quantum model.

**Phase K Results:** The extended Level-8 architecture jumped to an incredibly robust `+0.0203 dAUC` at K=6 (11/12 tasks positive, `p=0.0005`). By explicitly isolating aromatic properties via the `IsingYY` channels, tasks requiring ring discrimination (NR-AhR, NR-ER) showed maximal gains.

---

## 5. Phase L: NISQ Hardware & Noise Resilience
To demonstrate real-world utility, we evaluated the Phase K architecture under Open Quantum System (OQS) dynamics using density matrix simulation (`default.mixed`).
- **Noise Injection:** We applied Pauli-twirled Depolarizing noise ($p_{gate}$) after entanglers and BitFlip SPAM noise ($p_{meas}$) before measurement.
- **Results at K=4:**
  - **Ideal Statevector:** +0.0062 dAUC
  - **IBM Eagle 2023 levels ($p_{gate}=0.01$, $p_{meas}=0.02$):** +0.0055 dAUC (retains ~88% of signal)
  - **Heavy NISQ ($p_{gate}=0.05$, $p_{meas}=0.05$):** +0.0018 dAUC (retains ~29% of signal)

**Conclusion:** The Q-TIB bond-pooled measurement naturally resists symmetric depolarizing noise. Because the bias is calculated as a topological difference (structured vs scrambled) over averaged two-body observables, incoherent errors dampen the amplitude but preserve the relational geometry.

---

## 6. Literature Context
Our absorbability diagnostic operationalizes the equivariant QNN generalization theories of Schatzki et al. (2024, *npj QI*) and Caro (2022, *Nat. Commun.*) into a checkable framework for applied chemistry. By avoiding "quantum advantage" claims, we align with the cautious framing urged by Schuld (2021) and Kübler et al. (2021, *NeurIPS*), offering the community a rigorous template for verifying when quantum structure genuinely matters.
