import os
import argparse
import functools
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon
from torch_geometric.loader import DataLoader
from src.train import Trainer
from src.quantum_levels import *
from src.baselines import *
from src.data_loader import get_or_build_merged_dataset

_CLASSICAL = {
    1: Level1Classical, 2: Level2Classical, 3: Level3Classical, 4: Level4Classical,
    5: Level5Classical, 6: Level6Classical, 7: Level7Classical,
}
_QUANTUM = {
    1: Level1Quantum, 2: Level2Quantum, 3: Level3Quantum, 4: Level4Quantum,
    5: Level5Quantum, 6: Level6Quantum, 7: Level7Quantum,
}


def get_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@functools.lru_cache(maxsize=None)
def _quantum_param_count(level, scale, layers, num_tasks):
    """Param count of a quantum level. Cached: independent of fold/seed."""
    return get_param_count(_QUANTUM[level](hidden_dim=64, n_qubits=scale, q_layers=layers,
                                           out_dim=num_tasks, ansatz='strong'))


@functools.lru_cache(maxsize=None)
def match_classical_inner_dim(target_params, level, num_tasks, in_dim=64):
    """Smallest classical inner_dim whose param count best matches target_params.
    Cached so the (expensive) search runs once per (level, target), not per fold."""
    best_inner, min_diff = 1, float('inf')
    for inner in range(1, 513):
        params = get_param_count(_CLASSICAL[level](hidden_dim=in_dim, out_dim=num_tasks, inner_dim=inner))
        diff = abs(params - target_params)
        if diff < min_diff:
            min_diff, best_inner = diff, inner
        if params > target_params and diff > min_diff:
            break
    return best_inner


# Quantum model variants that share gates/parameters/depth and differ only in the
# chemistry->operator-geometry mapping:
#   quantum   = structured  (the proposed inductive bias)
#   scrambled = same circuit, mapping destroyed (isolates the inductive bias itself)
#   separable = same single-qubit ops, entanglement removed (isolates entanglement)
_ANSATZ = {'quantum': 'strong', 'scrambled': 'scrambled', 'separable': 'separable'}


def create_model(level, scale, layers, num_tasks, m_type):
    if m_type in _ANSATZ:
        return _QUANTUM[level](hidden_dim=64, n_qubits=scale, q_layers=layers,
                               out_dim=num_tasks, ansatz=_ANSATZ[m_type])
    if m_type == 'classical':
        q_params = _quantum_param_count(level, scale, layers, num_tasks)
        inner = match_classical_inner_dim(q_params, level, num_tasks, 64)
        return _CLASSICAL[level](hidden_dim=64, out_dim=num_tasks, inner_dim=inner)
    raise ValueError(m_type)


def per_task_auc(y_true, y_prob):
    """Per-task ROC-AUC over pooled CV predictions (NaN where not computable)."""
    aucs = np.full(y_true.shape[1], np.nan)
    for t in range(y_true.shape[1]):
        valid = ~np.isnan(y_true[:, t])
        yt, yp = y_true[valid, t], y_prob[valid, t]
        if len(np.unique(yt)) > 1:
            try:
                aucs[t] = roc_auc_score(yt, yp)
            except ValueError:
                pass
    return aucs


def paired_task_test(auc_a, auc_b):
    """Paired Wilcoxon across tasks (high power: hundreds of paired tasks, unlike the
    n=5 fold-level test whose minimum achievable two-sided p is 0.0625). Returns
    (p_value, median_delta, n_tasks) where delta = auc_a - auc_b per task."""
    mask = ~np.isnan(auc_a) & ~np.isnan(auc_b)
    if mask.sum() < 1:
        return 1.0, 0.0, 0
    delta = auc_a[mask] - auc_b[mask]
    median_delta = float(np.median(delta))
    if np.allclose(delta, 0):
        return 1.0, median_delta, int(mask.sum())
    try:
        p = wilcoxon(auc_a[mask], auc_b[mask]).pvalue
    except ValueError:
        p = 1.0
    return float(p), median_delta, int(mask.sum())


def bootstrap_metrics(y_true, y_prob, n_resamples=500, n_tasks_sample=20, seed=0):
    rng = np.random.default_rng(seed)
    roc_aucs, pr_aucs = [], []
    n_samples = len(y_true)
    n_tasks = y_true.shape[1]
    for _ in range(n_resamples):
        idx = rng.integers(0, n_samples, n_samples)
        yt_b, yp_b = y_true[idx], y_prob[idx]
        tasks = rng.choice(n_tasks, min(n_tasks_sample, n_tasks), replace=False)
        r, p = [], []
        for i in tasks:
            valid = ~np.isnan(yt_b[:, i])
            yt, yp = yt_b[valid, i], yp_b[valid, i]
            if len(np.unique(yt)) > 1:
                try:
                    r.append(roc_auc_score(yt, yp))
                    p.append(average_precision_score(yt, yp))
                except ValueError:
                    pass
        if r:
            roc_aucs.append(np.mean(r))
            pr_aucs.append(np.mean(p))
    if not roc_aucs:
        return (0.5, 0.5), (0.0, 0.0)
    return (np.percentile(roc_aucs, 2.5), np.percentile(roc_aucs, 97.5)), \
           (np.percentile(pr_aucs, 2.5), np.percentile(pr_aucs, 97.5))


def compute_pos_weight(tr_ds, num_tasks, device):
    all_labels = torch.cat([data.y for data in tr_ds], dim=0)
    pos_weight = []
    for t in range(num_tasks):
        valid = ~torch.isnan(all_labels[:, t])
        pos = all_labels[valid, t].sum().item()
        neg = valid.sum().item() - pos
        pos_weight.append(neg / (pos + 1e-5))
    return torch.tensor(pos_weight, dtype=torch.float32).to(device)


def parse_args():
    p = argparse.ArgumentParser(
        description="Quantum inductive-bias benchmark: structured vs scrambled (bias control) "
                    "vs separable (entanglement control) vs parameter-matched classical, "
                    "on scaffold-grouped CV")
    p.add_argument('--levels', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument('--qubits', type=int, nargs='+', default=[4, 6])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--datasets', type=str, nargs='+', default=['Tox21', 'ToxCast'])
    p.add_argument('--no_cache', action='store_true', help='Re-featurize instead of using the disk cache')
    p.add_argument('--out', type=str, default='results/benchmark_cv_results.csv')
    p.add_argument('--quick', action='store_true',
                   help='Fast smoke run: levels 1-3, 4 qubits, 3 folds, 20 epochs')
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.levels, args.qubits, args.folds, args.epochs, args.patience = [1, 2, 3], [4], 3, 20, 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device} | levels={args.levels} qubits={args.qubits} "
          f"folds={args.folds} epochs<={args.epochs} (patience {args.patience})")

    cache_path = None if args.no_cache else os.path.join('data', f"featurized_{'_'.join(args.datasets)}.pt")
    dataset, num_tasks = get_or_build_merged_dataset(root_dir='.', datasets=tuple(args.datasets), cache_path=cache_path)
    num_tasks = int(num_tasks)

    # Scaffold-grouped CV: each Bemis-Murcko scaffold lands entirely in one fold, so
    # the test fold is structurally novel (a deployment-relevant OOD split) rather than
    # the over-optimistic random split. GroupKFold guarantees groups never cross folds.
    scaffolds = list(getattr(dataset, 'scaffolds', []))
    if len(scaffolds) != len(dataset):
        raise RuntimeError("dataset.scaffolds missing/misaligned; cannot run scaffold CV")
    uniq_scaffolds = {s: i for i, s in enumerate(sorted(set(scaffolds)))}
    groups = np.array([uniq_scaffolds[s] for s in scaffolds])
    gkf = GroupKFold(n_splits=args.folds)
    print(f"Scaffold CV: {len(dataset)} molecules across {len(uniq_scaffolds)} scaffolds, "
          f"{args.folds} folds")

    all_results = []
    # quantum=structured (headline), scrambled=inductive-bias control,
    # separable=entanglement control, classical=context baseline.
    model_types = ['classical', 'separable', 'scrambled', 'quantum']

    for level in args.levels:
        for scale in args.qubits:
            print(f"\n--- Level {level} (Qubits: {scale}) ---")
            fold_rocs = {m: [] for m in model_types}
            fold_prs = {m: [] for m in model_types}
            fold_briers = {m: [] for m in model_types}
            pooled_probs = {m: [] for m in model_types}
            pooled_trues = {m: [] for m in model_types}
            indices = np.arange(len(dataset))

            for fold, (train_idx, test_idx) in enumerate(gkf.split(indices, groups=groups)):
                # Carve validation off the training scaffolds (kept scaffold-disjoint from
                # train so early stopping isn't tuned on leaked structures).
                rng = np.random.default_rng(42 + fold)
                tr_groups = np.unique(groups[train_idx])
                rng.shuffle(tr_groups)
                n_val_groups = max(1, int(0.1 * len(tr_groups)))
                val_group_set = set(tr_groups[:n_val_groups].tolist())
                is_val = np.array([g in val_group_set for g in groups[train_idx]])
                val_idx = train_idx[is_val]
                train_idx = train_idx[~is_val]

                tr_ds, va_ds, te_ds = dataset[train_idx], dataset[val_idx], dataset[test_idx]
                tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
                va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
                te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)
                pos_weight = compute_pos_weight(tr_ds, num_tasks, device)

                for m_type in model_types:
                    # Same init seed across model types within a fold -> reproducible & comparable.
                    torch.manual_seed(1000 * fold + 7)
                    model = create_model(level, scale, args.layers, num_tasks, m_type)
                    trainer = Trainer(model, device, pos_weight)

                    best_val_loss, no_improve = float('inf'), 0
                    for ep in range(args.epochs):
                        trainer.train_epoch(tr_loader)
                        va_mets = trainer.evaluate(va_loader)
                        trainer.scheduler.step(va_mets['loss'])
                        if va_mets['loss'] < best_val_loss:
                            best_val_loss, no_improve = va_mets['loss'], 0
                        else:
                            no_improve += 1
                            if no_improve >= args.patience:
                                break

                    te_mets = trainer.evaluate(te_loader)
                    fold_rocs[m_type].append(te_mets['roc_auc'])
                    fold_prs[m_type].append(te_mets['pr_auc'])
                    fold_briers[m_type].append(te_mets['brier'])
                    pooled_probs[m_type].append(te_mets['y_prob'])
                    pooled_trues[m_type].append(te_mets['y_true'])
                    print(f"  fold {fold+1}/{args.folds} {m_type:10s} "
                          f"test macro-ROC {te_mets['roc_auc']:.4f} PR {te_mets['pr_auc']:.4f} "
                          f"Brier {te_mets['brier']:.4f} (stopped ep {ep+1})")

            for m in model_types:
                pooled_probs[m] = np.vstack(pooled_probs[m])
                pooled_trues[m] = np.vstack(pooled_trues[m])

            # --- Significance ---
            # The HEADLINE test is structured (quantum) vs scrambled: same gates, params,
            # depth and entanglement, differing only in the chemistry->operator mapping. A
            # gain here is attributable to the inductive bias itself, not to capacity.
            # separable isolates entanglement; classical is a context baseline.
            comparisons = ['scrambled', 'separable', 'classical']
            n_comp = len(comparisons)

            # Fold-level Wilcoxon (low power: with n=5 folds the smallest possible
            # two-sided p is 0.0625, so it can essentially never reach 0.05). Reference only.
            def fold_p(a, b):
                try:
                    return min(1.0, wilcoxon(a, b).pvalue * n_comp)
                except ValueError:
                    return 1.0

            # Per-task paired Wilcoxon over pooled CV predictions (hundreds of paired
            # tasks -> properly powered, directly tests "is structured better per task").
            auc = {m: per_task_auc(pooled_trues[m], pooled_probs[m]) for m in model_types}
            p_task, d_task, p_fold = {}, {}, {}
            for c in comparisons:
                p, d, n = paired_task_test(auc['quantum'], auc[c])
                p_task[c] = min(1.0, p * n_comp)   # Bonferroni over the 3 comparisons
                d_task[c] = d
                p_fold[c] = fold_p(fold_rocs['quantum'], fold_rocs[c])
                tag = {'scrambled': 'Scrambled  (BIAS)', 'separable': 'Separable  (ENT) ',
                       'classical': 'Classical  (CTX) '}[c]
                print(f"  Structured vs {tag}: per-task p={p_task[c]:.4g} "
                      f"(median dAUC {d_task[c]:+.4f}, {n} tasks) | fold p={p_fold[c]:.4g}")

            for m in model_types:
                ci_roc, ci_pr = bootstrap_metrics(pooled_trues[m], pooled_probs[m], n_resamples=args.bootstrap)
                row = {
                    'Level': level, 'Qubits': scale, 'Model': m,
                    'ROC_AUC_CV_Mean': float(np.mean(fold_rocs[m])),
                    'ROC_AUC_CV_Std': float(np.std(fold_rocs[m])),
                    'PR_AUC_CV_Mean': float(np.mean(fold_prs[m])),
                    'PR_AUC_CV_Std': float(np.std(fold_prs[m])),
                    'Brier_CV_Mean': float(np.mean(fold_briers[m])),
                    'Brier_CV_Std': float(np.std(fold_briers[m])),
                    'ROC_CI_95': ci_roc, 'PR_CI_95': ci_pr,
                }
                for c in comparisons:
                    row[f'p_task_vs_{c}'] = p_task[c] if m == 'quantum' else np.nan
                    row[f'median_dAUC_vs_{c}'] = d_task[c] if m == 'quantum' else np.nan
                    row[f'p_fold_vs_{c}'] = p_fold[c] if m == 'quantum' else np.nan
                all_results.append(row)

            # Incremental save so a long run is never lost.
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            pd.DataFrame(all_results).to_csv(args.out, index=False)

    print(f"\nBenchmark saved to {args.out}")


if __name__ == "__main__":
    main()
