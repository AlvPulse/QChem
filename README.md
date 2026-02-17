# Hybrid Quantum-Classical Graph Neural Network for Tox21 Toxicity Prediction

This repository implements a novel **Hybrid Ensemble VQC (Variational Quantum Classifier)** architecture designed for **Multi-Task Toxicity Prediction** on the Tox21 dataset. The model is specifically engineered for **NISQ (Noisy Intermediate-Scale Quantum)** devices, focusing on circuit depth reduction, robustness, and information efficiency.

## Key Features

### 1. "Random Forest of QNNs" Architecture
Instead of a single deep quantum circuit (which suffers from noise and vanishing gradients), we employ an **Ensemble of Shallow Quantum Estimators**.
*   **Mechanism:** The latent space from the classical GNN is split (or copied) and fed into multiple independent, shallow quantum circuits (2-4 qubits, 1-2 layers).
*   **Benefit:** Drastically reduces circuit depth while maintaining expressivity through ensembling. This is inherently more noise-resilient than standard VQCs.

### 2. Intelligent Information Compression (Supervised Contrastive Loss)
To maximize the utility of the limited qubit space, we implement a **Supervised Contrastive Loss** as an auxiliary objective.
*   **Goal:** Force the classical encoder to map molecules with similar toxicity profiles to similar regions in the quantum latent space *before* quantum processing.
*   **Benefit:** The quantum circuit receives a "clean," high-density representation, effectively acting as a high-performance classifier on optimized features.

### 3. Multi-Task Learning
The model predicts **all 12 Tox21 tasks** simultaneously.
*   **Handling Missing Data:** A custom `MaskedBCEWithLogitsLoss` is used to handle the sparse nature of the dataset (many molecules have labels for only a subset of tasks).
*   **Imbalance Handling:** Dynamic positive weighting is applied per task.

### 4. Hardware-Efficient Ansatz (HEA)
The quantum circuits utilize a hardware-efficient ansatz (Rotations + CNOT Ring) to minimize gate overhead, making the model suitable for near-term quantum hardware execution.

---

## Architecture Diagram

```
[Molecule Graph]
      |
[Classical GNN Encoder (GINE/GAT)] --> Extracts Structural Features
      |
[Projection Head (MLP)] --> Compresses to Latent Space (e.g., 16 dims)
      |
      +---> [Supervised Contrastive Loss] (Auxiliary Training Signal)
      |
[Split/Copy] --> [Chunk 1] [Chunk 2] [Chunk 3] [Chunk 4]
                    |         |         |         |
              [Q-Est 1] [Q-Est 2] [Q-Est 3] [Q-Est 4] (Shallow Quantum Circuits)
                    |         |         |         |
              [Linear]  [Linear]  [Linear]  [Linear]
                    |         |         |         |
                    +---------+---------+---------+
                              |
                        [Aggregation] --> [12 Task Predictions]
```

## Installation

Ensure you have Python 3.8+ installed.

```bash
pip install torch torch-geometric pennylane rdkit scikit-learn pandas numpy tqdm
```

## Running Experiments

Use the `run_experiment.py` script to train models and evaluate performance.

### 1. Train the Novel Hybrid Ensemble
```bash
python run_experiment.py --model hybrid_ensemble --estimators 4 --qubits 4 --layers 2 --alpha 0.1 --epochs 20
```

*   `--estimators`: Number of parallel quantum circuits.
*   `--qubits`: Number of qubits per estimator.
*   `--alpha`: Weight of the contrastive loss (0.0 to 1.0).
*   `--split`: Whether to split the latent vector chunks or copy the full vector to each estimator.

### 2. Baselines

**Random Forest on Morgan Fingerprints (Standard Cheminformatics Baseline):**
```bash
python run_experiment.py --model rf
```

**Classical Ensemble (to isolate Quantum Advantage):**
Runs a classical neural network with the exact same architecture as the Hybrid model, but replaces the Quantum Layers with MLPs of equivalent parameter count.
```bash
python run_experiment.py --model classical_ensemble --estimators 4
```

**Standard Classical GNN:**
```bash
python run_experiment.py --model classical_gnn --gnn gine
```

## Repository Structure

*   `src/run_experiment.py`: Main entry point.
*   `src/hybrid_model.py`: Contains `HybridEnsembleVQC`.
*   `src/quantum_layers.py`: Implements `QuantumEnsemble` and `QuantumLayer` (PennyLane).
*   `src/classical_gnn.py`: PyG implementation of GINE/GAT encoders.
*   `src/loss.py`: Custom Masked BCE and Contrastive Loss functions.
*   `src/data_loader.py`: Tox21 data processing and scaffold splitting.
*   `src/baselines.py`: Implementation of RF and Classical Ensemble baselines.

## Future Work
*   Deployment on real quantum hardware (e.g., IBM Quantum) via PennyLane plugins.
*   Integration of Error Mitigation techniques (ZNE).
