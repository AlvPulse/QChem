"""Pool the per-seed BIAS deltas from both multiseed logs and run proper paired tests.

Sign test (binomial) and Wilcoxon signed-rank are nonparametric and far more defensible at
small n than a normal-approx CI. Null hypothesis: structured and scrambled are exchangeable
(no graph-topology inductive bias), so each per-seed delta is equally likely +/-.
"""
import re, glob, numpy as np
from scipy.stats import wilcoxon, binomtest

deltas = []
for log in ['_alt_b_multiseed.log', '_alt_b_multiseed2.log']:
    try:
        txt = open(log).read()
    except FileNotFoundError:
        continue
    for m in re.finditer(r'BIAS ([+-]\d+\.\d+)', txt):
        deltas.append(float(m.group(1)))

d = np.array(deltas)
n = len(d)
npos = int((d > 0).sum())
print(f"pooled seeds: n={n}")
print(f"per-seed BIAS (structured - scrambled): {np.round(d, 4).tolist()}")
print(f"mean {d.mean():+.4f}  std {d.std(ddof=1):.4f}  "
      f"min {d.min():+.4f}  max {d.max():+.4f}")
print(f"structured > scrambled: {npos}/{n}")

# Sign test: P(>= npos positives | p=0.5), one-sided
sign = binomtest(npos, n, 0.5, alternative='greater')
print(f"sign test (one-sided): p = {sign.pvalue:.4g}")

# Wilcoxon signed-rank on deltas vs 0, one-sided 'greater'
try:
    w = wilcoxon(d, alternative='greater')
    print(f"Wilcoxon signed-rank (one-sided greater): W={w.statistic:.1f}, p = {w.pvalue:.4g}")
except ValueError as e:
    print(f"Wilcoxon: {e}")

# Normal-approx 95% CI for reference
se = d.std(ddof=1) / np.sqrt(n)
print(f"mean +/- 1.96*SE: [{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}]")
print("POOL_DONE")
