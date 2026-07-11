# Project Overview: Quantum Inductive Bias for Molecular Toxicity Prediction

## 1. Problem Statement

Predicting molecular toxicity is a central challenge in drug discovery and environmental safety. The Tox21 and ToxCast datasets together contain 629 binary assay outcomes across roughly 10,000 molecules. Standard deep learning approaches—graph neural networks (GNNs), random forests on fingerprints—treat this as a multi-label classification task.

This project asks a more focused research question:

> **Does encoding chemical domain knowledge directly into the geometry of a quantum circuit's operators produce a measurable inductive bias advantage over a structurally identical circuit that ignores that knowledge?**

This deliberately sidesteps the overreaching claim "quantum beats classical." Instead, it frames quantum circuits as a structured-encoding tool and tests whether *that structure* carries signal.

---

## 2. Core Hypothesis

Chemical structure has natural operator-level analogues in quantum mechanics:

| Chemical Feature | Quantum Operator |
|---|---|
| Local motifs / substructure | `R_y` rotation (local observable) |
| Ring / aromatic cycles | `R_z` rotation (phase operator) |
| Global spectral structure | `IsingXX` entanglement (interaction Hamiltonian) |
| 3D bond distances | Controlled `CRZ` (coupling strength ∝ 1/distance) |
| Electronegativity / charge | `R_z / R_y` amplitude |
| Bonding interactions | `IsingXX / IsingYY` |
| Pharmacophore geometry | `U3` full single-qubit rotation + `CRX/CRY` |

The hypothesis is that mapping chemistry onto its natural operator family—*structured encoding*—will outperform an otherwise identical circuit where that mapping is randomly destroyed—*scrambled encoding*.

---

## 3. Project Goal

Produce a rigorous, reproducible benchmark that:

1. Measures the performance gap between structured and scrambled quantum circuits across seven levels of increasing encoding specificity.
2. Confirms that entanglement adds beyond what separable single-qubit circuits provide.
3. Contextualises gains (or lack thereof) against a parameter-matched classical MLP baseline.
4. Uses statistically powerful methodology: scaffold-grouped cross-validation on 629 tasks with paired Wilcoxon testing.

---

## 4. Four Model Variants (Per Level)

Every level is instantiated as four models trained and evaluated identically:

| Variant | Description |
|---|---|
| **Quantum** | Full structured circuit — chemistry drives operator family and qubit assignment |
| **Scrambled** | Same circuit, gates, depth, parameter count. Fixed random permutations destroy the chemistry→qubit mapping |
| **Separable** | Same single-qubit rotations, entanglement gates removed entirely |
| **Classical** | Parameter-matched MLP with the same hidden dimensions |

The primary scientific signal is the `Quantum − Scrambled` delta, which isolates inductive bias with capacity held constant.

---

## 5. Seven-Level Progression

| Level | Theme | Key Encoding |
|---|---|---|
| 1 | Features → Models | Three generic circuits with attention aggregation; no chemistry→operator mapping |
| 2 | Features → Operator Families | Motifs→R_y, Cycles→R_z, Spectral→IsingXX |
| 3 | Features → Operator Geometry | Phase modulation: R_z(cycle + α·motif); richer readout |
| 4 | 3D Spatial Entanglement | CRZ(1/(1+dist)); closer atoms couple more strongly |
| 5 | Hückel / Electronic Structure | R_z/R_y for electronegativity; XX/YY for bonding |
| 6 | Electrostatic Mapping | All-to-all CRZ(chem_i · chem_j); full single-qubit RX/RY/RZ |
| 7 | Pharmacophore / Reactivity | U3 rotations per site; CRX/CRY for pharmacophore dependencies |
| **8** | **Measurement-based graph readout** | **Qubits = coarse graph nodes; bond-gated IsingXX + ⟨Z_iZ_j⟩/⟨X_iX_j⟩ correlators bond-pooled by real adjacency** |

Levels 1–7 build on each other, progressively tightening the correspondence between chemical
intuition and the circuit's *operator geometry*. **Level 8 is architecturally distinct** (see
`docs/04_inductive_bias_probe.md`): it treats qubits as molecular-graph nodes and puts the bias in
the *measurement* (which correlators are read, selected by the real bond adjacency) rather than in
gate routing. This is the only design here whose `structured − scrambled` control is provably
non-absorbable *and* scales with qubit count — it lives in `run_bias_probe.py` / `run_levelG_probe.py`,
not in `run_benchmark.py`'s seven abstract-qubit levels.

> **Control-validity note.** For the seven `run_benchmark.py` levels, `structured − scrambled` is
> only a valid inductive-bias test at Levels 5–7 (weakly at 3); it is **absorbable/vacuous at
> Levels 1, 2, 4** because each circuit input passes through a free trainable projection. Details
> and proof in `docs/04`.

---

## 6. Technology Stack

| Component | Library / Tool |
|---|---|
| Quantum circuits | PennyLane |
| Graph neural network encoder | PyTorch Geometric (GINEConv) |
| Molecular featurization | RDKit (3D conformers, Gasteiger charges, pharmacophores) |
| Dataset | PyG MoleculeNet (ToxCast) + local CSV (Tox21) |
| Training | PyTorch, AdamW, ReduceLROnPlateau |
| Statistics | SciPy (Wilcoxon), scikit-learn (ROC-AUC, AUPRC, Brier) |
| Scaffold split | Bemis-Murcko via RDKit |

---

## 7. Repository Layout

```
QChem/
├── run_benchmark.py          # Full 5-fold scaffold CV over all 7 levels
├── run_experiment.py         # Single train/val/test run
├── src/
│   ├── quantum_levels.py     # Core: 7 levels × 4 variants = 28 model classes
│   ├── data_loader.py        # Featurization, caching, scaffold split, merging
│   ├── train.py              # Multi-loss trainer, dual-LR optimizer
│   ├── loss.py               # MaskedBCE + SupervisedContrastive
│   ├── features/
│   │   └── semantic_extractor.py  # GINEConv encoder → Motif/Cycle/Spectral reps
│   └── models/               # Quantum kernel, older GNN variants
├── docs/                     # This documentation
└── results/                  # CSV outputs from benchmark runs
```

---

## 8. Scientific Context

This work sits at the intersection of:

- **Quantum machine learning (QML)**: Using parameterised quantum circuits (PQCs) as trainable function approximators.
- **Molecular property prediction**: Graph-based learning on molecular graphs with multi-task binary labels.
- **Inductive bias research**: The long-standing ML question of whether architecture encodes the right priors for a given domain.

The project does *not* claim quantum speedup, quantum advantage on near-term hardware, or that VQCs are generally superior to classical networks. The contribution is narrower and more defensible: a carefully controlled ablation showing whether chemistry-informed operator geometry produces measurably better toxicity predictions than a structurally identical but chemically uninformed circuit.
