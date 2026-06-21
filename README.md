# Hybrid Quantum-Classical Graph Neural Network for Tox21 Toxicity Prediction

This repository demonstrates the evolutionary development of a novel Quantum Machine Learning architecture for **Multi-Task Toxicity Prediction** on the Tox21 dataset. We step from basic Exploratory Data Analysis, through classical GNN baselines, and advance to deep Quantum Inductive Biases, showing a clear trajectory from "just using a quantum circuit" to "encoding chemical geometry into quantum operator geometry."

---

## Evolutionary Trajectory & Project Structure

The project is structured to demonstrate an incremental buildup:

### Step 1: Exploratory Data Analysis (EDA)
*   **Files:** `DatasetEDA.ipynb`, `EDA_dataset.csv`
*   **Description:** Initial analysis of the Tox21 dataset, covering class imbalance, task correlations, and the sparsity (NaN labels) present across the 12 tasks.

### Step 2: Feature Extraction Pipeline
*   **Files:** `src/data_loader.py`
*   **Description:** Processing SMILES strings into PyTorch Geometric graphs, extracting node (atom) and edge (bond) features, and applying scaffold splitting.

### Step 3: Classical End-to-End Baselines
*   **Files:** `src/classical_gnn.py`, `src/baselines.py`, `run_classical.py`
*   **Description:** Implementation of standard GINE/GATv2 networks and Random Forests (using Morgan Fingerprints) to establish a strong classical baseline for the multi-task problem.

### Step 4: Hybrid End-to-End VQC (The Initial Ansatz)
*   **Files:** `src/hybrid_model.py`, `src/quantum_layers.py`, `run_quantum.py`
*   **Description:** Our initial attempt at quantum advantage: A "Random Forest of QNNs" (Hybrid Ensemble VQC) using Supervised Contrastive Loss. While it works, it lacked a rigorous quantum inductive bias beyond simple feature chunking.

### Step 5: Quantum Kernel Methods & SVM
*   **Files:** `src/benchmark_sqk.py`, `run_kernel.py`
*   **Description:** Exploring Structured Quantum Kernels (SQK) to capture cross-modality interactions (spectral-modulated motif encoding) using SVMs, establishing the theoretical groundwork for operator geometry.

### Step 6: Advanced Multi-Task Coupling & Loss Functions
*   **Files:** `src/loss.py`
*   **Description:** Custom `MaskedBCEWithLogitsLoss` to seamlessly handle the missing labels in Tox21 across all models.

### Step 7: Deep Quantum Inductive Bias (7 Levels of Novelty)
*   **Files:** `src/quantum_levels.py`, `run_experiment.py`
*   **Description:** To demonstrate quantum advantage over classical networks of the *same parameter count*, we define seven structural levels. Each level has three variants that share the classical `SemanticFeatureExtractor` encoder and differ only in the head:
    *   `*_quantum` — full PennyLane variational circuit (with entanglement).
    *   `*_separable` — **ablation**: identical circuit with **all entangling gates removed**, isolating whether entanglement (not just extra parameters) drives any gain.
    *   `*_classical` — a parameter-matched classical MLP (inner width auto-tuned in `run_benchmark.py` to match the quantum parameter count).

    The levels:
    *   **Level 1 (Features -> Models):** "Three Circuits + Attention". Motif, Cycle, and Spectral features routed to independent circuits and aggregated by attention.
    *   **Level 2 (Features -> Operator Families):** "Chemical-to-Operator Correspondence". Motifs -> local $R_y$; Cycles -> $R_z$ phase; Spectral -> Ising $XX$ Hamiltonians.
    *   **Level 3 (Features -> Operator Geometry):** "Chemical Operator Geometry". Motif features modulate cycle phase accumulation ($R_z(c + \alpha m)$); spectral features directly dictate entanglement strength.
    *   **Level 4 (3D Spatial Entanglement):** Euclidean distances modulate $CRZ$ entanglement (closer atoms -> stronger coupling).
    *   **Level 5 (Electronic Structure / Hückel):** $R_z/R_y$ reflect electronegativity/partial charges; $XX/YY$ entanglement reflects bonding.
    *   **Level 6 (3D Electrostatic Mapping):** full single-qubit rotations + all-to-all $CRZ$ coupling weighted by spatial features.
    *   **Level 7 (Pharmacophore / Reactivity):** $U3$ rotations per reactivity site + controlled $CRX/CRY$ pharmacophore dependencies.

---

## Installation

Ensure you have Python 3.8+ installed.

```bash
pip install torch torch-geometric pennylane rdkit scikit-learn pandas numpy tqdm matplotlib seaborn scipy
```

## Datasets

Three options, selected with `--dataset`:

*   `tox21`   — Tox21, 12 tasks (parsed from `EDA_dataset.csv`).
*   `toxcast` — ToxCast, 617 tasks (auto-downloaded via PyG `MoleculeNet`).
*   `merged`  — **Tox21 + ToxCast combined into a single 629-task problem.** Molecules are keyed by canonical SMILES; a molecule absent from one dataset gets `NaN` labels for that block, which the masked BCE / contrastive losses ignore. Built by `build_merged_dataframe` in `src/data_loader.py`.

All datasets share the same featurization (`mol_to_pyg`): 5 categorical atom features, 8 continuous chemical features (`x_cont`: partial charge, electronegativity, 3D coords, pharmacophore tags), 3D conformers, and 6 auxiliary molecular descriptors. **Note:** 3D conformer embedding makes featurization of the merged set slow on first run.

## Running Experiments

Use `run_experiment.py` to train a single model. Add `--dataset merged` (or `tox21` / `toxcast`).

```bash
# Levels 1-7, quantum vs parameter-matched classical, on the merged dataset
python run_experiment.py --model level3_quantum   --dataset merged --qubits 4 --layers 2
python run_experiment.py --model level3_classical --dataset merged

# Older baselines
python run_experiment.py --model classical_gnn --gnn gine --dataset tox21
python run_experiment.py --model hybrid_ensemble --estimators 4 --qubits 4 --layers 2
python run_experiment.py --model rf
```

## Full Benchmark Suite

`run_benchmark.py` runs the rigorous comparison: stratified K-fold CV over the
selected levels × {quantum, separable, classical} × qubit scales on the **merged
Tox21 + ToxCast** dataset, with significance tests and bootstrap 95% CIs. Results
are written incrementally to `results/benchmark_cv_results.csv`.

```bash
python run_benchmark.py                 # full run (levels 1-7, qubits {4,6}, 5 folds)
python run_benchmark.py --quick         # fast smoke run (levels 1-3, 4 qubits, 3 folds, 20 epochs)
python run_benchmark.py --levels 3 5 --qubits 4 --folds 5 --epochs 100
```

Useful flags: `--levels`, `--qubits`, `--folds`, `--epochs`, `--patience`,
`--batch_size`, `--bootstrap`, `--datasets`, `--no_cache`, `--out`.

**Significance reporting.** Two tests are reported per configuration:
*   **Per-task paired Wilcoxon** (primary): pairs quantum vs classical/separable ROC-AUC across the hundreds of tasks (computed on identical pooled CV predictions). This is well powered.
*   **Fold-level Wilcoxon** (reference only): with 5 folds the smallest achievable two-sided p-value is 0.0625, so it can essentially *never* reach 0.05 — don't read significance into it.

Both are Bonferroni-corrected (×2) for the two comparisons. The CSV also reports `median_dAUC_vs_{classical,separable}` (median per-task AUC gap).

### Performance notes
The earlier version was dominated by two costs, both fixed:
*   **Featurization** (3D conformer embedding of ~10k molecules) now runs once and is cached to `data/featurized_<datasets>.pt` (delete it or pass `--no_cache` to rebuild).
*   **Quantum circuits** previously ran one molecule at a time through PennyLane; they are now **batched via parameter broadcasting** (~50× faster per forward), plus parameter-matching is memoized across folds.
