"""E2 -- Gradient-variance probe: barren-plateau test for TC-QIC bond-pooled readout.

Cerezo 2021: LOCAL cost -> Var[grad] = Omega(1/K) => var*K = O(1).
             GLOBAL cost -> Var[grad] = O(1/exp(K)) => var*K -> 0.
levelG (bond-pooled, sum of O(K) 2-local observables) should show var*K ~ const,
validating Master Theorem clause (iii).
"""
import os, json, argparse, numpy as np, torch

from run_bias_probe import featurize
from run_levelG_probe import GraphG, CONFIGS

# Quantum parameter groups to track individually
QKEYS = ('theta', 'ringp', 'pairp', 'enc')


def _grad_norms(model):
    """Return grad_norm_sq for each quantum param group + classical total."""
    out = {}
    for key in QKEYS:
        out[key] = sum(
            p.grad.pow(2).sum().item()
            for n, p in model.named_parameters()
            if key in n and p.grad is not None
        )
    out['quantum'] = sum(out[k] for k in QKEYS)
    out['classical'] = sum(p.grad.pow(2).sum().item() for n, p in model.named_parameters()
                           if not any(q in n for q in QKEYS) and p.grad is not None)
    out['total'] = out['quantum'] + out['classical']
    return out


def probe_one(cfg_name, cfg, k, QFt, ATt, n_mols, n_inits, base_seed):
    """n_inits random-init forward+backward passes; return mean/var of grad_norm_sq per group."""
    # skip classical configs (no quantum circuit)
    if cfg.get('kind') == 'classical':
        return None
    all_runs = {key: [] for key in list(QKEYS) + ['quantum', 'classical', 'total']}
    n = min(n_mols, QFt.size(0))
    for i in range(n_inits):
        torch.manual_seed(base_seed * 1000 + i)
        model = GraphG(k, entangler=cfg['entangler'], readout=cfg['readout'])
        # GraphG.__init__ already uses torch.randn -> random weights; no pretrained load needed
        model.train()
        idx = torch.randperm(QFt.size(0))[:n]
        logits = model(QFt[idx], ATt[idx])
        loss = logits.mean()   # simple scalar -- standard for BP probes; no labels needed
        model.zero_grad()
        loss.backward()
        gn = _grad_norms(model)
        for key in all_runs:
            all_runs[key].append(gn[key])
    stats = {}
    for key, vals in all_runs.items():
        arr = np.array(vals, dtype=float)
        stats[key] = dict(
            mean=float(arr.mean()),
            var=float(arr.var(ddof=1) if len(arr) > 1 else 0.0),
        )
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qubits',   type=int, nargs='+', default=[4, 6, 8])
    ap.add_argument('--n_inits',  type=int, default=20)
    ap.add_argument('--n_mols',   type=int, default=32)
    ap.add_argument('--datasets', type=str, nargs='+', default=['Tox21', 'ToxCast'])
    ap.add_argument('--configs',  type=str, nargs='+', default=['levelG', 'meas_only', 'gate'])
    ap.add_argument('--out',      type=str, default='results/grad_var.json')
    ap.add_argument('--seed',     type=int, default=42)
    args = ap.parse_args()

    cfg_names = [c for c in args.configs if c in CONFIGS]
    rows = []

    print(f"\n{'K':>3}  {'config':>10}  {'group':>10}  {'mean_gnorm_sq':>15}  {'var_gnorm_sq':>15}  {'var*K':>12}")
    print('-' * 80)

    for k in args.qubits:
        QF, AT, AR, Y, SCAF = featurize(k, args.datasets)
        QFt = torch.tensor(QF)
        ATt = torch.tensor(AT)   # structured adj for all configs (BP is a circuit property)

        for cfg_name in cfg_names:
            cfg = CONFIGS[cfg_name]
            print(f"  probing K={k} config={cfg_name} ({args.n_inits} inits) ...", flush=True)
            stats = probe_one(cfg_name, cfg, k, QFt, ATt, args.n_mols, args.n_inits, args.seed)
            if stats is None:
                continue

            row = dict(k=k, config=cfg_name, n_inits=args.n_inits, n_mols=args.n_mols)
            for grp, s in stats.items():
                row[f'mean_{grp}'] = s['mean']
                row[f'var_{grp}']  = s['var']
                row[f'var_x_K_{grp}'] = s['var'] * k
            rows.append(row)

            for grp in list(QKEYS) + ['quantum', 'total']:
                print(f"{k:>3}  {cfg_name:>10}  {grp:>10}  "
                      f"{row[f'mean_{grp}']:>15.4e}  "
                      f"{row[f'var_{grp}']:>15.4e}  "
                      f"{row[f'var_x_K_{grp}']:>12.4e}")

    print('\n==== BP VERDICT (var_quantum * K): O(1)=local / exponential-decay=global ====')
    print(f"  {'K':>3}  {'config':>10}  {'var_quantum':>14}  {'var*K':>12}")
    for r in rows:
        print(f"  {r['k']:>3}  {r['config']:>10}  "
              f"{r['var_quantum']:>14.4e}  {r['var_x_K_quantum']:>12.4e}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f'\nsaved {len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
