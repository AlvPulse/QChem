"""Figure-data harness for the benchmark report.

Reuses the cached coarse-graph featurization (data/bias_coarse_K*.npz) and the exact model
definitions from run_bias_probe.py / run_levelG_probe.py to reproduce the Level-8 family at a
MODEST fidelity while CAPTURING the per-task / pooled-prediction / training-curve data that the
report figures need (the published headline numbers in docs/05 come from higher-fidelity runs;
this harness produces corroborating, illustrative artifacts at lower fold/seed/epoch counts).

Outputs per (config, K) to results/figdata/:
  auc_s, auc_c        : (12,) per-task pooled-CV ROC-AUC, structured / scrambled
  pr_s,  pr_c         : (12,) per-task AUPRC
  brier_s, brier_c    : (12,) per-task Brier
  pool_s, pool_c, Y   : (N,12) pooled-CV test probabilities + labels (each molecule predicted once)
  curve_s, curve_c    : (epochs,) fold-0/seed-0 validation-ROC training curves
  run_deltas          : (seeds,) run-level structured-minus-scrambled dAUC
  ctx                 : {'separable':.., 'classical':..} one-fold context AUCs
"""
import os, json, argparse, numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import wilcoxon, binomtest

from run_bias_probe import (featurize, scaffold_folds, standardize, masked_bce, roc12,
                            pos_weight, per_task_auc, GraphQ, ClassicalRef, N_TASKS)
from run_levelG_probe import GraphG, CONFIGS

OUT = 'results/figdata'


def build_model(kind, k):
    if kind == 'classical':
        return ClassicalRef(k)
    if kind == 'separable':
        return GraphQ(k, 'separable')
    cfg = CONFIGS[kind]                       # gate / levelG / meas_only
    return GraphG(k, entangler=cfg['entangler'], readout=cfg['readout'])


def train_capture(kind, variant, k, seed, tr, va, te, QF, AT, AR, Y, epochs, batch=128, want_curve=False):
    torch.manual_seed(seed)
    model = build_model(kind, k)
    adj = AR if variant == 'scrambled' else AT
    QFt, At, Yt = torch.tensor(QF), torch.tensor(adj), torch.tensor(Y)
    pw = pos_weight(Y, tr)
    qkeys = ('theta', 'ringp', 'pairp', 'enc')
    opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if any(q in n for q in qkeys)], 'lr': 1e-2},
        {'params': [p for n, p in model.named_parameters() if not any(q in n for q in qkeys)], 'lr': 1e-3},
    ], weight_decay=1e-4)
    best_va, best_probs = -1.0, np.full((len(te), N_TASKS), np.nan)
    curve, tr_t = [], torch.as_tensor(tr)
    for _ in range(epochs):
        model.train(); o = torch.randperm(len(tr))
        for s in range(0, len(tr), batch):
            bi = tr_t[o[s:s + batch]]
            loss = masked_bce(model(QFt[bi], At[bi]), Yt[bi], pw)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va_roc = roc12(model(QFt[va], At[va]).numpy(), Y[va])
            if want_curve:
                curve.append(float(va_roc))
            if va_roc > best_va:
                best_va = va_roc
                best_probs = torch.sigmoid(model(QFt[te], At[te])).numpy()
    return best_probs, curve


def per_task_scores(Y, probs):
    roc = np.full(N_TASKS, np.nan); pr = np.full(N_TASKS, np.nan); bri = np.full(N_TASKS, np.nan)
    for t in range(N_TASKS):
        v = ~np.isnan(Y[:, t]) & ~np.isnan(probs[:, t])
        if len(np.unique(Y[v, t])) > 1:
            roc[t] = roc_auc_score(Y[v, t], probs[v, t])
            pr[t] = average_precision_score(Y[v, t], probs[v, t])
            bri[t] = brier_score_loss(Y[v, t], probs[v, t])
    return roc, pr, bri


def run_config(kind, k, datasets, folds, seeds, epochs, do_ctx):
    QF0, AT, AR, Y, SCAF = featurize(k, datasets)
    fold_list = list(scaffold_folds(SCAF, folds))
    N = len(Y)
    print(f"[{kind} K{k}] {N} mols, {folds}f x {len(seeds)}s x {epochs}ep", flush=True)
    run_deltas, pool_s_seed, pool_c_seed = [], [], []
    curve_s = curve_c = []
    ctx = {}
    for si, seed in enumerate(seeds):
        ps = np.full((N, N_TASKS), np.nan); pc = np.full((N, N_TASKS), np.nan)
        for fi, (tr, va, te) in enumerate(fold_list):
            QF = standardize(QF0, tr)
            want = (si == 0 and fi == 0)
            ps[te], cs = train_capture(kind, 'structured', k, seed, tr, va, te, QF, AT, AR, Y, epochs, want_curve=want)
            pc[te], cc = train_capture(kind, 'scrambled', k, seed, tr, va, te, QF, AT, AR, Y, epochs, want_curve=want)
            if want:
                curve_s, curve_c = cs, cc
            if do_ctx and not ctx and want:
                sep, _ = train_capture('separable', 'structured', k, seed, tr, va, te, QF, AT, AR, Y, epochs)
                cl, _ = train_capture('classical', 'structured', k, seed, tr, va, te, QF, AT, AR, Y, epochs)
                ctx = {'separable': float(np.nanmean(per_task_auc(Y[te], sep))),
                       'classical': float(np.nanmean(per_task_auc(Y[te], cl)))}
        pool_s_seed.append(ps); pool_c_seed.append(pc)
        run_deltas.append(float(np.nanmean(per_task_auc(Y, ps)) - np.nanmean(per_task_auc(Y, pc))))
        print(f"    seed{seed}: struct {np.nanmean(per_task_auc(Y, ps)):.4f} "
              f"scram {np.nanmean(per_task_auc(Y, pc)):.4f} dAUC {run_deltas[-1]:+.4f}", flush=True)
    # Average pooled probabilities across seeds (each molecule predicted once per seed-CV).
    pool_s = np.nanmean(np.stack(pool_s_seed), 0); pool_c = np.nanmean(np.stack(pool_c_seed), 0)
    roc_s, pr_s, bri_s = per_task_scores(Y, pool_s)
    roc_c, pr_c, bri_c = per_task_scores(Y, pool_c)
    m = ~np.isnan(roc_s) & ~np.isnan(roc_c); d = roc_s[m] - roc_c[m]
    npos = int((d > 0).sum()); n = len(d)
    sgn = float(binomtest(npos, n, 0.5, alternative='greater').pvalue) if n else float('nan')
    try:
        wil = float(wilcoxon(d, alternative='greater').pvalue)
    except ValueError:
        wil = float('nan')
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f'{kind}_K{k}.npz'),
             auc_s=roc_s, auc_c=roc_c, pr_s=pr_s, pr_c=pr_c, brier_s=bri_s, brier_c=bri_c,
             pool_s=pool_s, pool_c=pool_c, Y=Y, curve_s=np.array(curve_s), curve_c=np.array(curve_c),
             run_deltas=np.array(run_deltas))
    summ = dict(kind=kind, k=int(k), struct=float(np.nanmean(roc_s)), scram=float(np.nanmean(roc_c)),
                median_dauc=float(np.median(d)), npos=npos, n=n, sign_p=sgn, wilcoxon_p=wil,
                run_mean=float(np.mean(run_deltas)), run_deltas=[round(x, 4) for x in run_deltas],
                ctx=ctx)
    print(f"    -> median dAUC {summ['median_dauc']:+.4f} {npos}/{n} pos "
          f"signp {sgn:.3g} wilp {wil:.3g} | sep {ctx.get('separable')} cls {ctx.get('classical')}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=str, default='fast',
                    help="fast = the report's figure set (gate/levelG K4, levelG K6, +context)")
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--datasets', type=str, nargs='+', default=['Tox21', 'ToxCast'])
    args = ap.parse_args()

    # (kind, K, run-context?)  -- minimal set that backs every data-driven figure.
    jobs = [('gate', 4, True), ('levelG', 4, True), ('levelG', 6, True), ('gate', 6, False)]
    summary = []
    for kind, k, ctx in jobs:
        summary.append(run_config(kind, k, args.datasets, args.folds, args.seeds, args.epochs, ctx))
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print("DONE figdata", flush=True)


if __name__ == '__main__':
    main()
