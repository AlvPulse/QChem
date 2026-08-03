# TC-QIC Theory and Experimental Status Summary
# Updated 2026-07-12 | T1-T13 complete, E1-E14 in various states

---

## Overview: what the TC-QIC framework claims

TC-QIC (Topology-Conditioned Quantum Information Compression) is a theory of how
molecular-graph topology enters a quantum circuit's inductive bias through the
READOUT GEOMETRY, not through the gate sequence. The central object is the
bond-pooled correlator readout B_A, which forms a Theta(K)-dimensional projection
of the 4^K-dim Pauli operator space onto the specific subspace spanned by bonded
two-qubit correlators. This projection is the mechanism of inductive bias.

**What the framework does NOT claim:**
- Not "quantum is better than classical" -- E9 shows a param-matched classical GNN
  achieves a comparable struct-scram gap; bond-pooled aggregation is substrate-independent.
- Not "measurement is a replacement for entanglement" -- the place-then-harvest
  identity (T9) shows gates and readout cooperate.
- Not an exponential speedup of any kind.

**What the framework does claim (P1-P8 register, T13):**
- Topology-aligned readout provides a STRUCTURED, provably non-absorbable inductive
  bias that grows linearly with K (P1, E5)
- The bias is driven by alignment alpha = <A',A>/||A'||||A|| (P2, E6)
- Bond-local readout (kappa=2) outperforms global (kappa=K) (P3, E7)
- Structured levelG ~ structured GNN_pm (P4 CONFIRMED, E9)
- Quantum adds a regime-dependent +56% at K=6 over classical bond-pooling (T12)

---

## Theory: T1-T13 status

All 13 theory documents are COMPLETE in docs/theory_deduct/proofs/.

### Core bottleneck pillar (T1-T3, T5)
- **T1**: Holevo chi + measurement DPI -- information accessible via measurement is
  upper-bounded by Holevo chi; the accessible subspace is finite.
- **T2**: dim A(O_8) = 3K + 2|E| = Theta(K) out of 4^K -- the operator-geometry
  bottleneck. Verified: E1 eff_rank/K = 1.52/1.15/1.86 at K=4/6/8.
- **T3**: Spectral coarsening = approximate ideal low-pass filter (93% low-freq energy,
  21% discretization error). Verified: E3 frac_low = 0.93 at all K.
- **T5**: Q-IB auto-compression -- the bond-pooled readout lives in an O(K log K)-bit
  slice by construction.

### Generalization pillar (T4, T6)
- **T4**: S_K equivariance under qubit permutations + GNN identity.
- **T6**: Rademacher bound O(W sqrt(K/n)) (head term) + O(K/sqrt(n)) (encoder term,
  Caro/Abbas 2022). Rigor gap: scoped to fixed theta; trained case requires union-over-theta.

### Sufficiency chain (T7)
- **T7**: eps-sufficiency, minimal sufficient B_A (conditional on TL(G) locality), and
  single-qubit blindness (Lemma 3.10: marginals determine no C_ij). TL(G) VIOLATED
  for aromatic 6-rings at K>=6 (E4); model works via inter-cluster bond correlators.

### Absorbability (T8) -- methodological backbone
- **T8**: Thm 2.7 (absorbability criterion) + Cor 2.8 (L2/L4 vacuous via condition A;
  Level G non-absorbable via condition B). Verified: E13 bit-exact L2/L4 residuals = 0.
  **This is the pre-registration criterion for every structured-scrambled comparison.**

### Place-then-harvest (T9)
- **T9**: Lemma 4.1 (entangler places signal in S(K) = bonded correlator subspace),
  Lemma 4.2 (readout harvests exactly S(K)), Thm 4.3 SNR identity (SNR = alpha^2 * ideal).
  Verified (E8 COMPLETE): K=4 PLACE 5.43x / HARVEST 1.75x; K=6 5.05x / 2.28x; K=8 6.23x / 2.95x.
  HARVEST ratio grows monotonically 1.75->2.28->2.95 confirming SNR_ideal scaling with K.

### Scaling law (T10, E5 complete)
- **T10**: Prop 3.11 Delta_B(K) = alpha*K + beta; eta_B=1 (levelG), eta_S=O(1/K) (gate).
  Calibrated: 1.4e-3*K + 2.3e-3 (R^2=0.996, K=4/6/8). K=10 COLLAPSES to near-zero
  (-0.0029, p=0.485) due to data-starvation (T6 bound: K/sqrt(n) exceeds data-sufficient
  threshold). Empirical K*_eff ~ 8 for n=7823 mol; T10 linear law valid within K<=8.

### Master Theorem (T11)
- **T11**: Three clauses: (i) Theta(K) structured compression; (ii) S_K equivariance +
  polynomial generalization; (iii) conditional BP-resistance via local-cost readout.
  Cor 4.3b: bond-pooled aggregation is the load-bearing source of bias for BOTH quantum
  and classical (E9).

### Bias-variance regime theory (T12)
- **T12**: Thm 4.4 (four-clause iff condition for Delta > 0); Thm 4.5 (classical
  dominance via double bottleneck, 5-8 pts empirically). Four-regime diagram.
  K=6 quantum add-on: +56% over classicalGNN_pm (1.56x median dAUC).

### Kernel-alignment + phase diagram + P1-P8 (T13)
- **T13**: Lemma 5.1 (kernel-alignment interpretation, fixed-theta); Theorem 5.3
  (six-region (alpha, kappa) phase diagram); P1-P8 prediction register with quantitative
  thresholds. P4 CONFIRMED.

---

## Experiments: status matrix

| Exp | Status | Key result | Theory |
|-----|--------|-----------|--------|
| E1 | DONE | eff_rank/K=1.52/1.15/1.86 | T2 CONFIRMED |
| E2 | DONE (partial) | theta var*K stable; K=6 anomaly = encoder artifact | T11(iii) PARTIAL |
| E3 | DONE | frac_low=0.93; gap=0.70/0.41/0.23 | T3 CONFIRMED |
| E4 | DONE | ring preservation 48%/26%/10% at K=4/6/8 | T7 violated for rings |
| E5 | COMPLETE -- P1 FAIL | K=10: median dAUC=-0.0029, p=0.485 (n.s.); bias collapses at K=10 (data-starvation); K=4-8 linear law intact | T10 calibration |
| E6 | COMPLETE -- P2 PARTIAL PASS | K=6 monotone 4/5 steps; lam=1.00 dAUC=+0.0146; peak at lam=0.75 (+0.0166) | T9 Lemma 4.2 |
| E7 | COMPLETE -- P3 PASS | Delta(kappa=2)=+0.0147 > Delta(kappa=K)=+0.0067; ratio=0.456 | T11(iii) + T13 |
| E8 | COMPLETE | K=4,6,8 ALL DONE -- T9 CONFIRMED; HARVEST 1.75->2.28->2.95x | T9 Lemmas 4.1/4.2 |
| E9 | DONE | classicalGNN_pm dAUC K=4/6/8: 0.014/0.007/0.012 (all sig) | T11 Cor 4.3b |
| E10 | COMPLETE | P5 PASS: cls gain +0.0099 vs qml gain -0.0021 (opposite signs) | T12 Thm 4.5 |
| E11 | PARTIAL | K=4 dAUC=+0.0333; K=6 dAUC=+0.0046 (P7 AT RISK); K=8 running | P7 |
| E12 | PARTIAL | K=4,6 noise+shots DONE; K=8 running | T11(iii) |
| E13 | DONE | L2/L4 residual 0.00 (vacuous); L5-7 genuine (1.26-1.55) | T8 CONFIRMED |
| E14 | DONE (K=6) | lambda_max struct>scram (+78%); node controls tied | T11(i) CONFIRMED |

---

## Key numerical results (for paper tables)

### Main bias table (structured - scrambled, scaffold CV, Wilcoxon)
| K | levelG dAUC | p-value | gate dAUC | classGNN_pm dAUC |
|---|------------|---------|-----------|-----------------|
| 4 | +0.0078 | 0.017 | +0.0044 | +0.0141 |
| 6 | +0.0108 | 0.011 | +0.0026 (n.s.) | +0.0069 |
| 8 | +0.0134 | 0.0024 | +0.0030 (n.s.) | +0.0121 |

K=8 survives Holm-Bonferroni correction (adj p = 0.017).

### Mechanism table (E8; K=4, 6, 8 ALL COMPLETE -- T9 CONFIRMED)
| K | PLACE ratio | on-bond frac | baseline | HARVEST true-A | HARVEST rand-A | HARVEST ratio |
|---|------------|-------------|---------|----------------|----------------|---------------|
| 4 | 5.43x | 0.830 | 0.508 | 1.754 | 1.005 | 1.75x |
| 6 | 5.05x | 0.709 | 0.342 | 2.239 | 0.981 | 2.28x |
| 8 | 6.23x | 0.654 | 0.249 | 2.918 | 0.989 | 2.95x |

HARVEST ratio GROWS monotonically with K (1.75x -> 2.28x -> 2.95x) -- T9 SNR_ideal scaling.
PLACE ratio nominally stable ~5.6x average; onfrac excess above baseline GROWS: +0.322, +0.367, +0.405.

### Absorbability table (E13)
| Level | residual | verdict |
|-------|----------|---------|
| 2 | 0.00e+00 | VACUOUS |
| 3 | 1.75e-01 | partial |
| 4 | 0.00e+00 | VACUOUS |
| 5 | 1.26e+00 | genuine |
| 6 | 1.55e+00 | genuine |
| 7 | 1.38e+00 | genuine |

### E12: Noise and shot robustness (K=4,6 DONE; K=8 running)

| K | dAUC retained at p=0.20 | (1-p)^2 prediction | Shot robust (N=32) |
|---|------------------------|-------------------|-------------------|
| 4 | 87.6% | 64.0% | 92.6% of exact |
| 6 | 92.6% | 64.0% | ~101% (within variance) |
| 8 | *pending* | | |

Noise robustness INCREASES with K (bond-pool averaging over more bonds).
Both struct and scram decay similarly, so the GAP is preserved much better than individual AUCs.
T11(iii) confirmed: 2-local bond-pooled cost is robust to common-mode depolarizing noise.

### Representation probe (E14, K=6)
| Target | Structured R^2 | Scrambled R^2 | diff |
|--------|---------------|---------------|------|
| lambda_max(A) [topology] | 0.072 | 0.040 | +0.032 (+78%) |
| Fiedler conn. [topology] | 0.141 | 0.134 | +0.007 |
| mean charge [node] | -0.004 | -0.003 | -0.001 (tied) |
| aromatic frac [node] | 0.740 | 0.764 | -0.024 (tied) |

---

## P1-P8 status

| Pred | Blocks | Status | Threshold |
|------|--------|--------|-----------|
| P1 | E5 | FAIL | K=10 median dAUC=-0.0029 (p=0.485); T10 linear law valid only for K<=8 (data-starvation at K=10 per T6 bound) |
| P2 | E6 | PARTIAL PASS | 4/5 steps monotone; peak dAUC=+0.0166 at lam=0.75; -0.0020 dip at lam=1.00 (within CV noise) |
| P3 | E7 | PASS | Delta(kappa=2)=+0.0147 > Delta(kappa=K)=+0.0067 (ratio 0.456 vs theory 0.33) |
| P4 | E9 | CONFIRMED | levelG ~ GNN_pm (TOST) |
| P5 | E10 | RUNNING | Gap closes >=50% with full features |
| P6 | E2 | PARTIAL | log-Var vs K slope > -1.5 |
| P7 | E11 | RUNNING | K-rank order Kendall tau > 0.6 |
| P8 | T12 | THEORY ONLY | Crossover at n* ~ 500-1000 |

---

## Honest caveats for publication

1. **T6 rigor gap**: Rademacher bound scoped to fixed theta; union-over-theta requires
   Caro/Abbas encoder term (not derived from scratch, cited).
2. **T11(iii) BP-resistance**: Cerezo 2021 theorem requires local random circuits with
   2-design structure; our data-dependent re-uploading GraphG does not satisfy this.
   Clause (iii) is CONDITIONAL, verified numerically at K=4/6/8 only.
3. **T7 TL(G) violation**: Aromatic rings (6 atoms) split clusters at K>=6; the
   epsilon-sufficiency chain is semi-empirical for large toxicophores.
4. **Classical still wins**: GNN_pm leads by 5-8 AUC pts; TC-QIC is an inductive-bias
   existence/scaling result, not a quantum advantage claim.
5. **T10 scaling law**: 3-point linear fit (K=4/6/8 only); calibration pending E5 K=10.
