# E13: Absorbability Audit — Empirical Confirmation of T8

**Status:** DONE
**Script:** `_verify_absorb.py`
**Theory connection:** T8 (Thm 2.7 + Cor 2.8)

---

## Protocol

For each benchmark level, construct structured layer `s` and scrambled layer `x` with
**identical variational weights** (copy_vars). Feed `s` the raw input; feed `x` the
input pre-permuted by the first gate-permutation applied to each input slot.
If scrambling = a re-labeling of free projection columns, the output is bit-exact equal
(residual = 0). A nonzero residual means the same projected vector is consumed under
*multiple inconsistent permutations* — a genuine constraint the network cannot absorb.

Setting: qubits=4, batch=8, seed=0.

---

## Results

| Lvl | residual     | verdict                | reuse note                               |
|-----|-------------|------------------------|------------------------------------------|
| 1   | (identity)  | sanity baseline        | no chem->operator routing; always equal  |
| 2   | **0.00e+00** | **ABSORBABLE (vacuous)** | m,c,s: 1 perm each                      |
| 3   | 1.75e-01    | partial (motif reuse)  | m: 2 perms (RY+phase); c,s: 1            |
| 4   | **0.00e+00** | **ABSORBABLE (vacuous)** | chem,dist: 1 perm each (both free proj) |
| 5   | 1.26e+00    | genuine                | chem: 4 perms (RZ,RY,XX,YY)             |
| 6   | 1.55e+00    | genuine                | chem: 5 perms (RX,RY,RZ,coupling)       |
| 7   | 1.38e+00    | genuine                | chem: 5 perms (U3x3,CRX,CRY)            |

---

## Interpretation

**T8 Cor 2.8 CONFIRMED (Levels 2 and 4).**
Bit-exact zero residual for Levels 2 and 4 means: any delta-AUC observed between
structured and scrambled at those levels is optimization noise, not inductive bias.
The scramble is fully re-absorbable by permuting rows of the free linear projections.
These levels provide *no* evidence of topology-driven inductive bias.

**Levels 5–7: genuine (non-absorbable).**
Residuals 1.26–1.55 at Levels 5–7 confirm multi-perm reuse prevents absorption.
The same chemistry vector is projected into gate parameters for multiple *inconsistent*
unitary slots; no single re-labeling of projection columns resolves all of them.

**Level 3: partial.**
The motif channel `m` is consumed by two gates (RY + phase), giving residual 0.175.
This is non-trivially non-absorbable but weaker than Levels 5–7 (only one channel
has multi-perm reuse; the chemistry and scaffold channels remain 1-perm each).

**Level G (GraphG, not in this script).**
Level G absorbability is established analytically in T8 via condition B: the adjacency
matrix `A` is per-molecule data (not a fixed permutation of a weight matrix), so the
scramble (random `AR`) cannot be absorbed by any re-labeling of static weights.
The empirical correlate is the consistently significant Wilcoxon p-values at Level G
across K=4/6/8 (main benchmark, E5).

---

## Methodological note

The absorbability audit is the methodological backbone for interpreting the main
benchmark. Only levels with genuine non-absorbable scrambles can provide evidence of
inductive bias. Levels 2 and 4 are vacuous controls; their delta-AUC cannot be cited
as bias evidence. Level G's non-absorbability (via condition B) is what makes the
structured-vs-scrambled gap scientifically meaningful.
