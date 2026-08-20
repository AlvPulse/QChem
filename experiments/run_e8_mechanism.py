"""E8: Place-then-harvest mechanism probe across K=4, 6, 8.

Extends make_mechanism.py (K=6 only) to K=4, 6, 8 to validate T9 Lemmas 4.1/4.2
at different qubit counts. Outputs results/mechanism_K{k}.npz for each K.

Run: python run_e8_mechanism.py --qubits 4 6 8
"""
import os, argparse, numpy as np, torch

from run_bias_probe import featurize, scaffold_folds, standardize, masked_bce, pos_weight, pairs_of
from run_levelG_probe import GraphG, CONFIGS


def run_mechanism_k(k):
    print(f"\n[K={k}] Place-then-harvest mechanism probe", flush=True)
    QF0, AT, AR, Y, SCAF = featurize(k, ["Tox21", "ToxCast"])
    tr, va, te = next(scaffold_folds(SCAF, 3))
    QF = standardize(QF0, tr)
    QFt, ATt, ARt, Yt = (torch.tensor(QF), torch.tensor(AT), torch.tensor(AR), torch.tensor(Y))
    PAIRS = pairs_of(k)
    pi = np.array([i for i, j in PAIRS]); pj = np.array([j for i, j in PAIRS])

    torch.manual_seed(0)
    model = GraphG(k, **CONFIGS["levelG"])
    pw = pos_weight(Y, tr)
    qkeys = ("theta", "ringp", "pairp", "enc")
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if any(q in n for q in qkeys)], "lr": 1e-2},
        {"params": [p for n, p in model.named_parameters() if not any(q in n for q in qkeys)], "lr": 1e-3},
    ], weight_decay=1e-4)
    tr_t = torch.as_tensor(tr)
    for ep in range(12):
        model.train(); o = torch.randperm(len(tr))
        for s in range(0, len(tr), 128):
            bi = tr_t[o[s:s + 128]]
            loss = masked_bce(model(QFt[bi], ATt[bi]), Yt[bi], pw)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 4 == 0:
            print(f"  K={k} ep{ep+1} loss {loss.item():.4f}", flush=True)
    model.eval()

    te_t = torch.as_tensor(te)
    with torch.no_grad():
        a = torch.atan(model.feat(QFt[te_t]))
        out = model.circ(a[:, :, 0], a[:, :, 1], ATt[te_t],
                         model.theta, model.ringp, model.pairp, model.enc)
        out = [o.float() for o in out]
    P = len(PAIRS)
    z = torch.stack(out[2 * k:3 * k], -1).numpy()
    zz = torch.stack(out[3 * k:3 * k + P], -1).numpy()
    Cconn = np.abs(zz - z[:, pi] * z[:, pj])

    A_te = AT[te]; A_pair = A_te[:, pi, pj]; AR_pair = AR[te][:, pi, pj]
    bonded = A_pair > 0

    on  = np.array([Cconn[m, bonded[m]].mean()  if bonded[m].any()  else np.nan for m in range(len(te))])
    off = np.array([Cconn[m, ~bonded[m]].mean() if (~bonded[m]).any() else np.nan for m in range(len(te))])
    on_m, off_m = np.nanmean(on), np.nanmean(off)
    mass_on = (Cconn * bonded).sum(1); mass_total = Cconn.sum(1)
    onfrac = np.nanmean(mass_on / (mass_total + 1e-9))
    base   = np.nanmean(bonded.sum(1) / Cconn.shape[1])

    def harvest(Aw):
        num = (Aw * Cconn).sum(1)
        den = Aw.sum(1) * (Cconn.sum(1) / Cconn.shape[1]) + 1e-9
        return np.nanmean(num / den)
    h_true = harvest(A_pair); h_rand = harvest(AR_pair)

    os.makedirs("results", exist_ok=True)
    out_path = f"results/mechanism_K{k}.npz"
    np.savez(out_path, on=on, off=off, on_m=on_m, off_m=off_m,
             onfrac=onfrac, base=base, h_true=h_true, h_rand=h_rand)

    ratio = on_m / off_m if off_m > 0 else float('nan')
    print(f"  K={k} PLACE   bonded {on_m:.4f} vs non-bonded {off_m:.4f}  ({ratio:.2f}x)", flush=True)
    print(f"  K={k} PLACE   on-bond mass frac {onfrac:.3f} vs uniform {base:.3f}", flush=True)
    print(f"  K={k} HARVEST true-A {h_true:.3f} vs random-A {h_rand:.3f}", flush=True)
    return dict(k=k, on_m=float(on_m), off_m=float(off_m), ratio=float(ratio),
                onfrac=float(onfrac), base=float(base), h_true=float(h_true), h_rand=float(h_rand))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qubits', type=int, nargs='+', default=[4, 6, 8])
    args = ap.parse_args()

    rows = []
    for k in args.qubits:
        cache = f"results/mechanism_K{k}.npz"
        if os.path.exists(cache):
            print(f"[K={k}] Loading cached {cache}")
            d = np.load(cache)
            ratio = float(d['on_m']) / float(d['off_m']) if float(d['off_m']) > 0 else float('nan')
            row = dict(k=k, on_m=float(d['on_m']), off_m=float(d['off_m']), ratio=ratio,
                       onfrac=float(d['onfrac']), base=float(d['base']),
                       h_true=float(d['h_true']), h_rand=float(d['h_rand']))
        else:
            row = run_mechanism_k(k)
        rows.append(row)

    print("\n" + "=" * 65)
    print("E8 MECHANISM SUMMARY (T9 Lemma 4.1/4.2 validation)")
    print(f"{'K':>4} | {'on_m':>8} | {'off_m':>8} | {'ratio':>7} | {'onfrac':>7} | {'h_true':>7} | {'h_rand':>7}")
    print("-" * 65)
    for r in rows:
        print(f"{r['k']:>4} | {r['on_m']:>8.4f} | {r['off_m']:>8.4f} | "
              f"{r['ratio']:>7.2f}x | {r['onfrac']:>7.3f} | {r['h_true']:>7.3f} | {r['h_rand']:>7.3f}")
    print()
    print("T9 pass criterion: ratio > 1x (bonded > non-bonded) AND h_true > h_rand")
    import json
    with open('results/e8_mechanism_summary.json', 'w') as f:
        json.dump(rows, f, indent=2)
    print("Saved -> results/e8_mechanism_summary.json")


if __name__ == '__main__':
    main()
