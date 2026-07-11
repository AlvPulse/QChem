# Deductive Action Plan: The Quantum-Topological Information Bottleneck (Q-TIB)

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
