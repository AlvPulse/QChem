# Benchmarking Setup

**Entry point**: `run_benchmark.py`

The benchmark is designed to answer one question with statistical rigor: does structured quantum encoding produce a measurable inductive bias advantage over a chemically uninformed circuit of the same capacity?

---

## 1. Datasets

### Tox21 (12 tasks)
- Source: `EDA_dataset.csv` (local file)
- ~8,000 molecules
- 12 binary toxicity assays (nuclear receptor and stress-response panel):
  `NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53`
- Class imbalance: ~1–5% positive per task
- Missing labels: present, handled by masked loss

### ToxCast (617 tasks)
- Source: PyG MoleculeNet (auto-downloaded)
- ~8,000 unique molecules
- 617 diverse toxicity endpoints
- High label sparsity: many molecules miss many tasks

### Merged Dataset (default, 629 tasks)
- Union of Tox21 and ToxCast molecules, keyed by canonical SMILES
- 629 tasks total (12 + 617)
- Molecules present in only one dataset get `NaN` labels for the other's tasks
- Reported in two **blocks**:
  - `Tox21` block: task indices 0–11 (headline)
  - `ToxCast` block: task indices 12–628 (reference)

---

## 2. Scaffold-Grouped Cross-Validation

### Why Scaffold Split?
Random splits allow scaffold overlap between train and test, inflating held-out AUC by 5–15 AUC points. A scaffold split ensures every held-out molecule has a structurally novel scaffold, simulating real deployment where the model encounters unseen chemical space.

### Implementation (`run_benchmark.py`, lines 219–226)
```python
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold

scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
uniq_scaffolds = {s: i for i, s in enumerate(sorted(set(scaffolds)))}
groups = np.array([uniq_scaffolds[s] for s in scaffolds])

gkf = GroupKFold(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, groups=groups)):
    ...
```

- Bemis-Murcko scaffolds computed per molecule via RDKit
- `GroupKFold` ensures each scaffold lands entirely in one fold
- Validation set carved from training scaffolds (not from the test fold)
- 5 folds, each producing a distinct ~20% held-out scaffold set

### Split Sizes (approximate)
| Fold | Train | Val | Test |
|------|-------|-----|------|
| Each | ~6,400 | ~800 | ~1,600 |

---

## 3. Model Variants Trained Per Configuration

For each `(level, n_qubits)` pair, four models are trained and evaluated:

| Variant | Label in CSV | Purpose |
|---|---|---|
| `quantum` | Full structured circuit | Headline model |
| `scrambled` | Same circuit, chemistry→qubit map destroyed | Controls for capacity |
| `separable` | Same rotations, entanglement removed | Controls for entanglement value |
| `classical` | Parameter-matched MLP | Contextual baseline |

> ⚠️ **Control validity (read `docs/04_inductive_bias_probe.md`).** The `scrambled` control is a
> fixed input permutation in front of a free `nn.Linear`, which training **re-absorbs** unless the
> same projected vector is reused under multiple conflicting permutations. Verified bit-exact
> (`_verify_absorb.py`):
> - **L1**: no chemistry→operator routing — `scrambled` ≡ `structured` (not a control).
> - **L2, L4**: each input has one permutation via its own free projection — **absorbable, VACUOUS**
>   (`structured − scrambled` is uninterpretable, ≈ optimisation noise).
> - **L3**: weak (only the motif's dual routing survives).
> - **L5, L6, L7**: one `chem` vector reused under 4–5 permutations — **genuine** controls.
>
> So treat `structured − scrambled` as a real inductive-bias signal **only at Levels 5–7** (weakly
> at L3). For a non-absorbable, *scalable* graph-topology bias, use the measurement-based **Level 8**
> probe (`run_bias_probe.py` / `run_levelG_probe.py`), documented in `docs/04`.

### Parameter Matching (`run_benchmark.py`, lines 31–69)
The classical model's hidden dimension is solved so its total parameter count equals the full quantum model's count:
```python
q_count = _quantum_param_count(level, n_qubits, n_layers, num_tasks)
inner_dim = match_classical_inner_dim(q_count, level, num_tasks)
```
`@functools.lru_cache` memoises the computation across folds.

---

## 4. Training Configuration

### Default Arguments (`run_benchmark.py`, lines 176–200)
| Parameter | Default | Notes |
|---|---|---|
| `--levels` | [1, 2, 3, 4, 5, 6, 7] | Which levels to run |
| `--qubits` | [4, 6] | Qubit counts to sweep |
| `--folds` | 5 | Scaffold CV folds |
| `--epochs` | 100 | Max epochs per fold |
| `--patience` | 15 | Early stopping epochs |
| `--layers` | 2 | Quantum circuit depth |
| `--batch_size` | 128 | |
| `--bootstrap` | 500 | Resamples for 95% CI |
| `--datasets` | ['Tox21', 'ToxCast'] | Merged by default |
| `--lr` | 1e-3 | Base LR (encoder, head) |
| `--q_lr` | 1e-2 | Quantum parameter LR |

### Quick Mode (`--quick` flag)
Reduces to: `levels=[1,2,3], qubits=[4], folds=3, epochs=20, patience=8`. Used for rapid iteration.

### Optimiser Details
- **AdamW** with `weight_decay=1e-4`
- **Two parameter groups**:
  - Quantum keys (`q_layer`, `q_motif`, `q_cycle`, `q_spectral`): LR = 1e-2
  - All other parameters: LR = 1e-3
- **ReduceLROnPlateau**: `factor=0.5, patience=5` on validation loss
- **Early stopping**: validation loss plateau for 15 epochs

---

## 5. Metrics

### Per-Task Metrics
For each of the 629 tasks independently:
- **ROC-AUC**: Area under the ROC curve. Tasks with only one class in the test fold are skipped.
- **AUPRC**: Area under the Precision-Recall curve (average precision). Better than ROC-AUC for severely imbalanced tasks.
- **Brier score**: Mean squared error between predicted probability and binary label. Measures calibration.

### Per-Block Aggregation
Metrics are aggregated separately within the Tox21 block and the ToxCast block:
```
ROC_Tox21_Mean, ROC_Tox21_Std    # mean/std of per-task ROC-AUC over 12 tasks
ROC_ToxCast_Mean, ROC_ToxCast_Std
PR_Tox21_Mean, PR_ToxCast_Mean
Brier_Tox21_Mean, Brier_ToxCast_Mean
```

### Bootstrap 95% Confidence Intervals (`run_benchmark.py`, lines 136–161)
```python
for b in range(500):
    idx_mols  = np.random.choice(n_samples, n_samples, replace=True)
    idx_tasks = np.random.choice(n_tasks, 20, replace=False)
    roc_b = compute_roc(preds[idx_mols][:, idx_tasks], labels[idx_mols][:, idx_tasks])
    ...
ci_lo, ci_hi = np.percentile(boot_rocs, [2.5, 97.5])
```
Each resample draws molecules with replacement and 20 random tasks, producing a distribution over ROC-AUC. The 2.5th and 97.5th percentiles form the reported interval.

---

## 6. Statistical Testing

### Primary Test: Per-Task Paired Wilcoxon (`run_benchmark.py`, lines 336–383)

**Unit of observation**: mean CV AUC per task (629 observations).

```python
from scipy.stats import wilcoxon

diffs = quantum_per_task_auc - scrambled_per_task_auc   # shape: (629,)
stat, p = wilcoxon(diffs, alternative='greater')
```

- **Why paired**: Each task sees the same molecules; pairing removes between-task variance.
- **Why Wilcoxon**: Non-parametric; AUC differences are not Gaussian, and some tasks are near-chance.
- **Power**: N=629 tasks gives high power even for small effects (delta AUC ~0.01 is detectable).

**Bonferroni correction**: Factor ×3 applied because three comparisons are made:
- quantum vs scrambled
- quantum vs separable
- quantum vs classical

### Secondary Test: Fold-Level Wilcoxon
**Unit of observation**: mean AUC per fold (5 observations).

Same test, but with N=5 folds. The minimum achievable two-sided p-value with N=5 is 0.0625, so this test is reported as "reference only" and cannot reach conventional significance thresholds. It is included for completeness and to check whether fold-level trends are consistent with task-level findings.

### Reported Significance Columns (per block)
```
p_task_{Block}_vs_scrambled       # primary, Bonferroni-corrected
median_dAUC_{Block}_vs_scrambled  # median of 629 per-task differences
p_fold_{Block}_vs_scrambled       # reference, low power

p_task_{Block}_vs_separable
median_dAUC_{Block}_vs_separable
p_fold_{Block}_vs_separable

p_task_{Block}_vs_classical
median_dAUC_{Block}_vs_classical
p_fold_{Block}_vs_classical
```

---

## 7. Output Format

Results accumulate in `results/benchmark_cv_results.csv`. The file is written incrementally after each `(level, n_qubits)` configuration, preserving partial results if a run is interrupted.

### CSV Columns (per row)
```
Level              # 1–7
Qubits             # 4 or 6
Model              # quantum / scrambled / separable / classical
PrimaryBlock       # Tox21 or ToxCast (first block)

# Per-block performance
ROC_Tox21_Mean, ROC_Tox21_Std
ROC_ToxCast_Mean, ROC_ToxCast_Std
PR_Tox21_Mean, PR_ToxCast_Mean
Brier_Tox21_Mean, Brier_ToxCast_Mean
ROC_{Block}_CI95           # "[lo, hi]" for primary block
PR_{Block}_CI95

# Significance (only populated for Model == 'quantum')
p_task_Tox21_vs_scrambled,  median_dAUC_Tox21_vs_scrambled,  p_fold_Tox21_vs_scrambled
p_task_Tox21_vs_separable,  median_dAUC_Tox21_vs_separable,  p_fold_Tox21_vs_separable
p_task_Tox21_vs_classical,  median_dAUC_Tox21_vs_classical,  p_fold_Tox21_vs_classical
# (repeated for ToxCast block)
```

---

## 8. Benchmark Invocation Examples

```bash
# Full benchmark (all 7 levels, 4 and 6 qubits, 5-fold scaffold CV)
python run_benchmark.py

# Quick test run
python run_benchmark.py --quick

# Single level, specific qubit count
python run_benchmark.py --levels 2 3 --qubits 4 --folds 3

# Tox21 tasks only (not merged)
python run_benchmark.py --tasks Tox21 --levels 1 2 3

# Increase bootstrap samples for tighter CIs
python run_benchmark.py --bootstrap 1000
```

---

## 9. Evaluation Philosophy

The benchmark is deliberately conservative:

1. **Scaffold split** rather than random split — penalises scaffold-memorising models.
2. **Parameter matching** — classical baseline is not a straw man; it has the same parameter budget.
3. **Primary comparison is quantum vs scrambled**, not quantum vs classical — this controls for model capacity and isolates the encoding mapping.
4. **Bonferroni correction** — reduces false-positive risk from multiple comparisons.
5. **AUPRC alongside ROC-AUC** — for tasks with <2% positive rate, ROC-AUC can be high even for a model that ignores positives; AUPRC penalises this.
6. **Both Tox21 and ToxCast reported separately** — prevents cherry-picking the more favourable block.
