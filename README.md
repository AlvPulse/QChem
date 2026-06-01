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

### Step 7: Deep Quantum Inductive Bias (The 3 Levels of Novelty)
*   **Files:** `src/quantum_levels.py`, `run_experiment.py` (New implementations)
*   **Description:** To truly demonstrate quantum advantage over classical networks of the same parameter count, we introduce three structural levels comparing Classical MLPs against Quantum Circuits:
    *   **Level 1 (Features -> Models):** "Three Circuits + Attention". Motif, Cycle, and Spectral features are routed to independent models and aggregated. (Weak novelty).
    *   **Level 2 (Features -> Operator Families):** "Chemical-to-Operator Correspondence". Motifs map to local $R_y$ observables; Cycles to $R_z$ phase operators; Spectral to $XY$ Hamiltonians. (Stronger novelty).
    *   **Level 3 (Features -> Operator Geometry):** "Chemical Operator Geometry". Deep cross-modulation where, for example, motif features dynamically modulate the phase accumulation of cycle features ($R_z(c + \alpha m)$), and spectral features dictate entanglement strength ($e^{-i \theta(s) Z_i Z_j}$). (Highest novelty).

---

## Installation

Ensure you have Python 3.8+ installed.

```bash
pip install torch torch-geometric pennylane rdkit scikit-learn pandas numpy tqdm
```

## Running Experiments

Use the `run_experiment.py` script to train models and evaluate performance across the evolutionary levels.

### Running the New Levels (Levels 1-3)

**Level 1:**
```bash
python run_experiment.py --model level1_quantum
python run_experiment.py --model level1_classical
```

**Level 2:**
```bash
python run_experiment.py --model level2_quantum
python run_experiment.py --model level2_classical
```

**Level 3:**
```bash
python run_experiment.py --model level3_quantum
python run_experiment.py --model level3_classical
```

### Running Older Baselines
```bash
# Standard Classical GNN
python run_experiment.py --model classical_gnn --gnn gine

# Hybrid Ensemble (Step 4)
python run_experiment.py --model hybrid_ensemble --estimators 4 --qubits 4 --layers 2

# Random Forest
python run_experiment.py --model rf
```
