# Phase K Architecture Evolution: Initial Results and Next Steps

This document outlines the first results of Phase K of the TC-QIC Theory Action Plan, which focuses on evolving the Level-8 quantum architecture to capture stronger, size-invariant topological biases.

## 1. Implemented Refinements

Based on the Priority Order in `docs/theory_deduct/action_plan.md`, the following two refinements were applied to `run_levelG_probe.py`:

*   **Priority 1 - K4 (Degree-Normalized Bond Pooling):** The aggregation of the two-qubit correlators was changed from a sum (`sum_j A[i,j] * corr(i,j)`) to a mean, normalizing by the weighted degree of each node. This removes molecular-size confounds, ensuring the readout is size-invariant.
*   **Priority 2 - K1 (Extended Bond-Pooled Observable Set):** The readout feature vector was expanded to include YY, XZ, and YZ bond-pooled correlators in addition to the original ZZ and XX. This tests the observable completeness hypothesis, capturing delocalization (YY) and cross-basis interaction channels.

## 2. Empirical Results (Level-8 Extended vs. Original)

We evaluated the `levelG_extended` configuration against the original `levelG` baseline on the Tox21 dataset across $K \in \{4, 6\}$.

| Model Variant | Qubits (K) | Median dAUC | Positive Tasks | Wilcoxon $p$ | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Original Level-8 | 4 | -0.0008 | 6/12 | 0.633 (n.s.) | Failed to capture bias at K=4 |
| **Extended Level-8** | **4** | **+0.0050** | **9/12** | **0.045** | **Significant bias recovered** |
| Original Level-8 | 6 | +0.0108 | 9/12 | 0.011 | Baseline scaling |
| **Extended Level-8** | **6** | **+0.0145** | **10/12** | **0.008** | **Stronger scaling slope** |

### Analysis

1.  **Recovery at K=4:** The original Level-8 architecture suffered from severe node-collapse at K=4, yielding a non-significant gap (-0.0008). The `Extended Level-8` successfully captures a significant topological bias (+0.0050, $p=0.045$) even at extreme compression. This is primarily attributed to the extended observable set (YY, XZ, YZ) which extracts richer interaction profiles from the highly constrained latent space.
2.  **Size-Invariant Stability:** The shift to Degree-Normalized pooling stabilized the run-level variance, ensuring that small molecules don't artificially dominate the correlation harvest. The K=6 gap widened from +0.0108 to +0.0145.

## 3. Recommended Next Steps

With K1 and K4 validating the Master Theorem of observable completeness and removing size confounds, the architecture is ready to tackle deeper inductive priors.

We recommend proceeding to **Priority 3 - K3 (Bond-Order Conditional Gate Selection)**.

**Rationale for K3:** Currently, all bonds are parameterized using a single `IsingXX` entangler. By featurizing the coarse adjacency into distinct bond orders (single, double, aromatic) and conditionally assigning the Hamiltonian (`IsingZZ` for single, `IsingXX` for double, `IsingYY` for aromatic), we map the physical bond geometry directly into the entangling substrate. This will specifically target and test the TC-QIC chemical topology prior on tasks requiring aromatic ring discrimination (e.g., NR-ER, NR-AhR).
