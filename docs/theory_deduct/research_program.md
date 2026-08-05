# TC-QIC RESEARCH PROGRAM -- Two-Phase Plan

---
## COMPLETION STATUS (updated 2026-07-12)

### Theory tasks
| Task | Status | File |
|------|--------|------|
| T1   | DONE   | proofs/T1_holevo_qib.md |
| T2   | DONE   | proofs/T2_operator_geometry.md |
| T3   | DONE   | proofs/T3_spectral_lowpass.md |
| T4   | DONE   | proofs/T4_equivariance_gnn.md |
| T5   | DONE   | proofs/T5_qib_lagrangian.md |
| T6   | DONE (scoped) | proofs/T6_rademacher_bound.md |
| T7   | DONE (conditional/semi-empirical) | proofs/T7_sufficiency_chain.md |
| T8   | DONE   | proofs/T8_absorbability.md |
| T9   | DONE   | proofs/T9_place_harvest.md |
| T10  | DONE   | proofs/T10_scaling_law.md |
| T11  | DONE   | proofs/T11_master_theorem.md |
| T12  | DONE   | proofs/T12_bias_variance.md |
| T13  | DONE   | proofs/T13_kernel_phase_predictions.md |

### Experimental tasks
| Task | Status | File |
|------|--------|------|
| E1   | DONE   | proofs/E1_operator_geometry_results.md + results/probe_opgeom.json |
| E2   | DONE (partial support) | proofs/E2_gradient_variance_results.md + results/grad_var.json |
| E3   | DONE   | proofs/E3_spectral_lowpass_results.md + results/e3_spectral.json |
| E4   | DONE (TL(G) violated for aromatic rings) | proofs/E4_preservation_results.md + results/e4_preservation.json |
| E5   | RUNNING (K=10 full-scale; PID 2692) | K=4,6,8 DONE; K=10 full-scale launched; prelim frac=0.05 was small-n |
| E6   | RUNNING (K=6 alpha sweep; job bquz6wp4z) | just started; results pending |
| E7   | COMPLETE -- P3 PASS | K=6: Delta(kappa=2)=+0.0147 > Delta(kappa=K)=+0.0067; ratio=0.456; results/e7_kappa_summary.json |
| E8   | COMPLETE (K=4,6,8 ALL DONE) | proofs/E8_mechanism_results.md; T9 CONFIRMED; HARVEST grows 1.75->2.28->2.95x |
| E9   | DONE (existing data) | proofs/E9_classical_gnn_analysis.md + cls_pm_K468.json |
| E10  | COMPLETE -- P5 PASS | cls gain +0.0099 vs qml -0.0021; results/e10_summary.json |
| E11  | PARTIAL (K=4,6 DONE; K=8 running; job b1tlx5vof) | BBBP K=4 dAUC=+0.0333; K=6 dAUC=+0.0046 (P7 AT RISK) |
| E12  | PARTIAL (K=4+6 DONE; K=8 running; job ba7z4jhfz) | proofs/E12_shots_noise_results.md; robustness increases with K |
| E13  | DONE | proofs/E13_absorbability_results.md |
| E14  | DONE (K=6) | proofs/E14_representation_probe_results.md + results/probe_K6.npz |

### Phase gate status
**PHASE GATE CLEARED (2026-07-12)**: T2 + T9 + Cor 2.8 all done.
E5 (K=10,12 scaling) background run is now unblocked.
E9 (param-matched classical GNN) is now unblocked -- highest priority experiment.

### MVTP (Minimum Viable Theoretical Package)
T1 + T2 + T4 + T6 (scoped) + T8 + T9 -- ALL COMPLETE.
The defensible core is ship-ready pending E5 and E9 experimental validation.

---

**Framing.** The theory rests on three provable pillars and one synthesis.
(1) *Operator-geometry bottleneck*: the readout projects a 4^K-dim operator space
onto a Theta(K) slice (Def 1.3, Lemma 1.1).
(2) *Topological/spectral bottleneck*: coarse-graining is an ideal graph low-pass
filter (Lemma 3.5).
(3) *Equivariance + generalization*: the bond-pooled readout is S_K-equivariant
with O(sqrt(K/n)) gap (Thm 2.4, 2.2).
The *Master Theorem* (4.3) and *scaling law* (3.11) synthesize these; they are
also the highest-risk claims and drive the load-bearing experiments (E5, E9).
The *Absorbability Theorem* (2.7/Cor 2.8) is the methodological backbone and is
already ~90% complete.

**Key implementation anchors.**
- Coarse-graining: `SpectralClustering(affinity='precomputed', assign_labels='discretize')`
  on bond adjacency in `run_bias_probe.py:coarse_graph`
- Level-8 model: `GraphG` in `run_levelG_probe.py` with configs
  `{gate, levelG, meas_only, classicalGNN, classicalGNN_pm(d=7/9/11)}`
- kappa-extension harness: `probe_entanglement_harvesting.py` (adds IsingZZ + YY)
- Figure scripts: `make_mechanism.py`, `make_shots.py`, `make_noise.py`, `make_probe.py`
- Absorbability: `_verify_absorb.py`

The prior phenomenological plan at `docs/theory_deduct/action_plan.md` (Q-TIB
Phases A-J) is superseded by this first-principles program; its Phase A/E/G/H map
onto E4/E8/E5/statistics respectively.

---

## PHASE 1: THEORETICAL FRAMEWORK

---

### T1: Quantum MI, Holevo ceiling, and the measurement DPI | Priority: HIGH | Effort: 4-6d | Deps: []

**Goal.** Make Lemma 1.1 rigorous: `I(X;T_O) <= chi <= I_Q(X;T)`, establishing
that *which observables are read* strictly caps extractable task information.

**Deliverables**
- [ ] Def 1.2 -- cq-state MI = Holevo chi
- [ ] Lemma 1.1 -- measurement bottleneck (DPI)
- [ ] Corollary I_acc <= chi

**Proof strategy.** Build rho_XT = sum_x p(x)|x><x| tensor rho_theta(x). MI =
D(rho_XT || rho_X tensor rho_T), monotone under id_X tensor M_O by
Lindblad-Uhlmann monotonicity of relative entropy. For a cq-state, direct entropy
computation gives I_Q = chi. Close with Holevo 1973.

**Risks.** The Level-8 readout is expectation-value estimation of signed Hermitian
Paulis + A-weighted sums, *not* a POVM, and the feature map is affine, not a
probability channel. The clean CPTP/entanglement-breaking story only holds if you
either (a) recast the readout as post-processing of the eigen-projector POVMs of
each O_a, or (b) drop the channel framing and bound accessible information directly
through the operator-subspace projection of T2. **Recommend (b) as primary, (a)
as the information-theoretic restatement.** Flag this in the paper; do not paper
over it.

---

### T2: Operator-geometry bottleneck -- accessible-subspace dimension | Priority: HIGH | Effort: 3-4d | Deps: []

**Goal.** Prove `dim A(O_8) = 3K + 2|E| = Theta(K)` out of 4^K, and that
trace-out is a hard hypothesis-class constraint (Prop 1.2), not noise.

**Deliverables**
- [ ] Def 1.3 with exact constants
- [ ] Prop 1.2 -- HS decomposition, lossless iff g*_{A_perp} = 0

**Proof strategy.** Pauli strings orthonormal under HS inner product. Read
operators = {X_i, Y_i, Z_i} (3K) union span{sum_j A_ij Z_iZ_j} union
span{sum_j A_ij X_iX_j}. Upper bound dim = 3K + rank(pool_Z) + rank(pool_X) <= 5K,
and subset span of on-bond correlators (3K+2|E|). Lower bound: generic linear
independence for a.e. adjacency. phi_O(rho) = <O_a, rho>_HS = <O_a, Pi_A rho>_HS
since O_a in A => dependence only through Pi_A rho.

**Risks.** Must distinguish *feature-vector dimension* (5K coordinates) from
*operator-subspace dimension* (3K+2|E|); pooling is a further rank-<=K linear map
so the two differ. "Generic independence" needs an explicit genericity
(measure-zero exception) statement.

---

### T3: Spectral coarse-graining as an ideal low-pass filter | Priority: MED | Effort: 4-5d | Deps: []

**Goal.** Prove Lemma 3.5 / Cor 3.6 / Def 3.4: spectral-clustering coarse-graining
= projection onto bottom-K Laplacian eigenspace = ideal graph low-pass; define
"high-frequency atomic noise" precisely.

**Deliverables**
- [ ] Lemma 3.5 -- ratio-cut relaxation -> Pi_{<=K} = sum_{k<=K} u_k u_k^T
- [ ] Cor 3.6 -- discarded band = high-graph-frequency node-field content
- [ ] Def 3.4 -- "high-frequency atomic noise" (formal)

**Proof strategy.** von Luxburg ratio-cut relaxation => bottom-K eigenvectors;
cluster-mean pooling = low-pass projection in the graph-Fourier (Laplacian) basis.

**Risks.** *Implementation mismatch.* `coarse_graph` runs
`SpectralClustering(affinity='precomputed', assign_labels='discretize')` on the
**bond adjacency A**, not the Laplacian; `discretize` is a nonlinear rounding.
So the exact "ideal low-pass" holds for the *relaxation*, and the discretized
cluster-means are only an *approximate* low-pass. State Lemma 3.5 for the
relaxation; bound the discretization gap or verify numerically (E3). Also
reconcile affinity=A vs L=D-A spectra.

---

### T4: Task symmetry, readout equivariance, GNN identity | Priority: HIGH | Effort: 3-4d | Deps: []

**Goal.** Axiom 3.2 + Prop 3.3 (orbit sufficiency), Thm 2.4 (S_K-equivariance),
Cor 2.5 (equivariant class + Rademacher dividend), Prop 2.6 (quantum
message-passing identity).

**Deliverables**
- [ ] Thm 2.4 (tightened) -- S_K-equivariance of bond-pooled readout
- [ ] Cor 2.5 with orbit-averaging factor
- [ ] Prop 2.6 explicit (psi, oplus) correspondence
- [ ] Axiom 3.2 + Prop 3.3 (orbit sufficiency via Fisher-Neyman)

**Proof strategy.** Conjugation identity Tr[O_i O_j U_pi rho U_pi^+] =
Tr[O_{pi^{-1}i} O_{pi^{-1}j} rho]; reindex. Prop 3.3 by Fisher-Neyman on the
quotient q: G -> [G]. Cor 2.5 via Elesedy-Zaidi equivariance generalization gain.

**Risks.** The full pipeline (spectral cluster -> arbitrary cluster->qubit
assignment -> circuit -> pool) is equivariant only up to a canonicalizing
relabeling; prove equivariance of the *pooled* output, not per-qubit b[i].
Quantitatively, molecular Aut(G) is usually near-trivial => the Rademacher
tightening ~= 1; the subset-F_equiv membership is real but the numeric dividend
is modest. Do not oversell.

---

### T5: The Q-IB Lagrangian synthesis | Priority: MED | Effort: 2-3d | Deps: [T1, T2]

**Goal.** Assemble Def 1.4 and prove the domain-restriction claim (classical IB
ranges over all channels; Q-IB over theta at fixed O) puts Q-IB automatically in
a strong-compression regime.

**Deliverables**
- [ ] Def 1.4 -- Q-IB Lagrangian
- [ ] Proposition "auto-compression": I(X;T_O) bounded by the Theta(K) slice

**Proof strategy.** Combine T1 ceiling + T2 dimension.

**Risks.** "I(X;T_O) <= Theta(K) * log(*)" is loose -- MI of a bounded d-dim
*continuous* feature is not <= dim without a resolution argument. Make precise via
metric-entropy/covering of the bounded feature cube or a rate-distortion
(Gaussian-channel d/2 * log(1+SNR)) bound. Otherwise this reads as hand-waving.

---

### T6: Rademacher generalization bound | Priority: HIGH | Effort: 5-7d | Deps: [T2, T4]

**Goal.** Prove Thm 2.2 (R(h) <= R_hat_n + O(W*sqrt(K/n)) + ...) and
Cor 2.3 (Omega(4^K)->O(K) sample-complexity saving).

**Deliverables**
- [ ] Lemma 2.1 -- B = Theta(sqrt(K))
- [ ] Thm 2.2 -- Rademacher bound
- [ ] Cor 2.3 -- sample-complexity saving Omega(4^K)->O(K)

**Proof strategy.** Bartlett-Mendelson linear-class bound R_hat_n <=
max||phi|| * W/sqrt(n); Lemma 2.1 from bounded degree; Talagrand contraction
for 1-Lipschitz loss.

**Risks.** *The single biggest rigor gap.* The clean bound holds for the
**linear head at fixed theta**. The real class is union_theta, whose complexity
includes the trained encoder. Must add the circuit capacity via a
Lipschitz-in-theta + covering-number argument over Theta (Caro et al. 2021/2022
QML generalization; Abbas et al. effective dimension). Either bound the union
properly or explicitly scope Thm 2.2 as "conditional on theta, + O(sqrt(dimTheta/n))
encoder term." This determines whether Cor 2.3 (the headline saving) is defensible.

---

### T7: Sufficiency chain | Priority: HIGH | Effort: 4-5d | Deps: [T2, T3]

**Goal.** Def 3.7 (epsilon-sufficiency), Thm 3.8 (conditional macro-topology
sufficiency), Thm 3.9 (B_A minimal sufficient), Lemma 3.10 (single-qubit
blindness).

**Deliverables**
- [ ] Def 3.7 -- epsilon-sufficiency (formal)
- [ ] Thm 3.8 -- conditional macro-topology sufficiency
- [ ] Thm 3.9 -- B_A minimal sufficient (with "conditional/semi-empirical" label)
- [ ] Lemma 3.10 -- single-qubit blindness

**Proof strategy.** Thm 3.8 via chain rule I(G;Y|C) = I(G;Y) - I(C;Y). Thm 3.9
Fisher-Neyman factorization through the A-contraction; minimality from off-bond
nullity. Lemma 3.10: S is a function of one-qubit marginals only; Theta(K^2)
correlators vs 3K.

**Risks.** Thm 3.8's hypothesis (toxicophore subset single cluster) is *empirical*
-> conditional theorem; verify by E4. Thm 3.9 minimality rests on the empirical
off-bond ~= 0 (0.013 vs 0.066) -> "minimal given the empirical placement,"
semi-empirical. Label both honestly; they are not unconditional.

---

### T8: Absorbability as function-class identity | Priority: HIGH | Effort: 2-3d | Deps: []

**Goal.** Promote Prop 1/Cor 2 to a general theorem: H_struct = H_scram under
data-independent reparametrization; non-absorbability iff (a) >=2 inconsistent
perms on one shared h, or (b) structure enters as fixed per-molecule data upstream
of the only trainable map. Certify Level 8 satisfies (b).

**Deliverables**
- [ ] Thm 2.7 -- general absorbability theorem
- [ ] Cor 2.8 -- (a)/(b) non-absorbability criterion
- [ ] Pre-registration criterion (trace each structured signal to the trainable map
  in front of it; check (a)/(b))

**Proof strategy.** theta = Wh; permutation P; W' = P^{-1}W is a bijection of
parameter space intertwining the families; independence across sites =>
simultaneous satisfiability. (b): the absorbing map would have to be
input-dependent (depend on A(mol)); a fixed Linear is not => no single W' works
for all molecules.

**Risks.** Lowest-risk task (already bit-exact for L2/L4 in `_verify_absorb.py`).
Only work: generalize (b) beyond Level 8 and phrase it as *equivariance-breaking
by data-dependent reparametrization*.

---

### T9: Place-then-harvest operator identity | Priority: HIGH | Effort: 3-4d | Deps: [T2]

**Goal.** Lemma 4.1 (entangler places correlation on bonded pairs), Lemma 4.2
(harvest overlap = <A',A>/||A'||||A||; max at A'=A, anti-aligned if <A',A><0).

**Deliverables**
- [ ] Lemma 4.1 -- place-on-bond with explicit 2-path leakage bound
- [ ] Lemma 4.2 -- harvest inner product identity

**Proof strategy.** Dyson/BCH expansion of U(A,theta) = prod exp(-i/2 A_ij theta_ij
X_iX_j); leading-order connected correlator C_ij prop A_ij theta_ij. Harvest inner
product sum_j A'_ij C_ij prop <A',A>.

**Risks.** "No correlation where A_ij=0" is **leading-order only**: higher-order
paths i-k-j create off-bond correlation at O(theta^2). State Lemma 4.1 with an
explicit 2-path leakage bound -- and note the empirical off-bond 0.013 (nonzero!)
vs on-bond 0.066 *confirms* the perturbative hierarchy rather than contradicting
it. This is a strength if framed correctly.

---

### T10: Bias scaling law | Priority: HIGH | Effort: 4-5d | Deps: [T6, T7, T9]

**Goal.** Derive Prop 3.11: Delta(K) prop eta_O(K) * s(K) - cW*sqrt(K/n);
Level 8 Theta(K) growth (eta_B=1), gate-only Theta(1) flat (eta_S=Theta(1/K));
predict saturation K*.

**Deliverables**
- [ ] Prop 3.11 -- bias scaling law (monotone, asymptotically linear until saturation)
- [ ] Alignment eta_O defined for each config
- [ ] Empirical LS realization Delta_B(K) ~= 1.4e-3*K + 2.3e-3 (R^2 ~= 0.996)
- [ ] K* prediction ~= 8-16 (heavy-atoms/fragment-size)

**Proof strategy.** eta_B = 1 since S(K) subset A(O_8); eta_S = Theta(1/K)
(marginals carry no connected correlator); net aligned mass Theta(K) vs Theta(1)
minus the T6 penalty.

**Risks.** *Over-claiming linearity.* The step "aligned signal mass Theta(K) =>
Delta AUC linear in K" links *accessible-signal dimension* to *realized AUC*
through an unknown monotone link. Three points (R^2=0.996) cannot distinguish
linear from mild-log/saturating. Present Prop 3.11 as "monotone, asymptotically
linear until saturation"; treat the exact slope as empirical calibration pending
E5 (K=10,12). Do not assert exact linearity as theory.

---

### T11: The Master Theorem (Thm 4.3) | Priority: HIGH | Effort: 5-6d | Deps: [T5, T6, T7, T9]

**Goal.** Assemble (i) topology-aligned tight compression, (ii) equivariance +
O(sqrt(K/n)) generalization, (iii) trainability via readout locality (no barren
plateau in the O(K) regime).

**Deliverables**
- [ ] Thm 4.3 clause (i) -- topology-aligned tight compression (from T1+T2+T9)
- [ ] Thm 4.3 clause (ii) -- equivariance + generalization (from T4+T6)
- [ ] Thm 4.3 clause (iii) -- trainability/BP-resistance (conditional on E2)
- [ ] Remark correcting "forbidden from Hilbert space" (unitary still explores 2^K;
  *measurement* projects onto Theta(K) slice; BP-resistance from readout locality
  + shallow depth)

**Proof strategy.** (i) = T1+T2+T9; (ii) = T4+T6; (iii) via Cerezo et al.
local-cost-function theorem: O(1) depth + <=2-local observables => Var[d_theta C]
= Omega(poly(1/K)), vs McClean global-observable 2^{-K}.

**Risks.** (iii) The Cerezo guarantee assumes specific block-local/2-design ansatz
structure; our gate is a **data-dependent** graph-gated IsingXX (re-uploading),
for which BP theory is less standard. Cite the correct result and **verify
numerically** (E2) rather than lean on the theorem alone. This clause must not
be stated as unconditionally proven.

---

### T12: Bias-variance regime theory | Priority: HIGH | Effort: 4-5d | Deps: [T6, T7, T10]

**Goal.** Thm 4.4 (strict-improvement iff alignment AND non-classical signal AND
capacity-match), Thm 4.5 (classical-dominance inequality ||g*_{A_perp}||^2 + eps >
O(W_cl*sqrt(cap/n))), and the derivation that the same double bottleneck both
cleans the bias and caps accuracy 5-8 pts; explain the *refuted* low-data prior
via the n* crossover.

**Deliverables**
- [ ] Thm 4.4 -- strict-improvement iff (alignment AND non-classical signal AND
  capacity-match)
- [ ] Thm 4.5 -- classical-dominance inequality
- [ ] n* analysis (Sec 5.1-5.2) -- crossover derivation

**Proof strategy.** Excess risk = approximation + estimation. TC-QIC:
approx = ||g*_{A_perp}||^2 + eps, estimation O(sqrt(K/n)); unconstrained classical:
approx ~= 0, estimation O(sqrt(cap_cl/n)). "Power of data" (Huang et al. 2021)
specialized to this architecture.

**Risks.** ||g*_{A_perp}||^2 and eps are not analytic; Thm 4.5 *accommodates and
predicts the sign/crossover* of the 5-8-pt lead but the *magnitude* is empirically
calibrated, not derived from scratch. State as a decomposition that predicts P8's
crossover behavior, not a first-principles "5-8."

---

### T13: Kernel-alignment view + phase diagram + prediction register | Priority: MED | Effort: 3-4d | Deps: [T9, T10, T11, T12]

**Goal.** Recast TC-QIC as quantum-kernel-target alignment (Sec 5.3); formalize
the (alpha, kappa) phase diagram (6 regions); compile P1-P8 as formal falsifiable
corollaries with quantitative pass/fail thresholds.

**Deliverables**
- [ ] Kernel-alignment lemma -- struct-scram gap = positive alignment increment
- [ ] Phase-diagram theorem -- (alpha, kappa) -> Delta sign/scaling
- [ ] P1-P8 prediction register with quantitative thresholds

**Proof strategy.** Schuld 2021 (QML = kernel), Kubler-Buchholz-Scholkopf
(alignment governs generalization); phase diagram maps alpha (from T9) and kappa
(from T7/Lemma 3.10) to Delta sign/scaling.

**Risks.** Kernel view is clean only for *fixed-theta* kernels; we **train** theta.
Either analyze the frozen-encoder kernel / quantum-NTK (lazy regime), or present
Sec 5.3 as an interpretive bridge, not a theorem.


---

## PHASE 1 DEPENDENCY GRAPH

```
T1 --+             T2 --+-------+--------+          T3 --+        T4 --+      T8 (standalone)
     |                  |       |        |               |             |
     +-- T5 (Def 1.4) --+       |        +-- T9 ---+     +-- T7 --+    |
          deps: T1,T2           |                  |    |        |     |
                                +-- T6 -------+    +----+-- T10 -+-- T12 --+
                                    deps: T2,T4    |                       |
                                              |    +-------- T11 ----------+
                                              |         deps: T5,T6,T7,T9  |
                                              +--- T13 (phase diagram, P1-P8)
                                                   deps: T9,T10,T11,T12
```

Roots (no deps): **T1, T2, T3, T4, T8**

Spine A (generalization): T2/T4 -> T6 -> T11/T12

Spine B (scaling): T2 -> T9 -> T10 -> T12 -> T13

Backbone: T8 (independent, ship anytime)

---

## PHASE 1 CLASSIFICATION TABLE

| Claim | Pure math | Needs computation | Note |
|---|---|---|---|
| Lemma 1.1 (DPI), Def 1.3 (dim), Prop 1.2 (HS) | yes | chi and I(X;T_O) values (E1) | dim analytic; MI measured |
| Thm 2.2 / Cor 2.3 (Rademacher) | yes | slope/constant of gap vs K,n (E5) | bound analytic; constant empirical |
| Thm 2.4 / Cor 2.5 (equivariance) | yes | -- | pure |
| Thm 2.7 / Cor 2.8 (absorbability) | yes | bit-exact residual (E13) | proof done; audit confirms |
| Lemma 3.5 (low-pass) | yes (relaxation) | discretization gap (E3) | discretize step is empirical |
| Thm 3.8 (epsilon-sufficiency) | yes conditional | eps and preservation rate (E4) | hypothesis is empirical |
| Thm 3.9 / Lemma 3.10 | yes / semi-empirical | off-bond nullity (E8) | minimality uses empirics |
| Prop 3.11 (K-law) | mechanism only | functional form + K* (E5) | linearity is empirical |
| Lemma 4.1/4.2 (place-harvest) | yes leading-order | 5.1x, 2.24x, alpha-sweep (E6, E8) | leakage measured |
| Master Thm clause (iii) trainability | conditional | Var[d_theta C] vs K,depth,kappa (E2) | must verify BP absence |
| Thm 4.4/4.5 (bias-variance) | decomposition | ||g*_{A_perp}||^2, eps, P4 gap (E9, E10) | magnitude empirical |

---

## PHASE 1 MATHEMATICAL TOOLS TO ACQUIRE

| Tool | For | Primary sources |
|---|---|---|
| Quantum DPI / relative-entropy monotonicity, Holevo bound | T1, T5 | Wilde QIT; Holevo 1973; Lindblad/Uhlmann |
| QML generalization (covering numbers, effective dimension) | T6 (union over theta) | Caro et al. 2021/2022; Abbas et al. 2021; Banchi et al. |
| Rademacher / equivariant generalization | T6, T4 | Bartlett-Mendelson 2002; Elesedy-Zaidi 2021 |
| Spectral graph theory / graph signal processing | T3 | von Luxburg 2007; Ortega et al. 2018 (GSP) |
| Information Bottleneck (classical + quantum) | T5, T7 | Tishby-Pereira-Bialek 1999; Salek/Datta quantum-IB |
| Sufficiency / Fisher-Neyman | T7, T9 | Lehmann-Casella |
| Barren-plateau theory (cost-locality) | T11(iii), P6 | McClean 2018; Cerezo 2021; Holmes 2022 |
| Quantum kernels / alignment / NTK | T13 | Schuld 2021; Kubler-Buchholz-Scholkopf 2021; Huang 2021 |
| Quantum information geometry (QFI, effective dim) | T6 optional | Meyer 2021; Abbas 2021 |
| MI estimation (KSG, MINE) | E1, E4 verification | Kraskov 2004; Belghazi 2018 |

---

## PHASE 1 LITERATURE REVIEW PLAN

**Bottleneck pillar (T1-T2, T5).** Quantum IB, Holevo, entanglement-breaking
channels -- position TC-QIC as *fixed-measurement* IB vs free-channel classical IB.

**Generalization pillar (T6, T4).** QML generalization bounds + geometric-deep-learning
equivariance -- the novelty is *equivariance from measurement geometry*, not
weight-tying.

**Scaling/trainability pillar (T10, T11).** Barren plateaus + expressibility --
contrast "deeper is better" with our locality-driven trainability.

**Comparative pillar (T12, T13).** Power-of-data, quantum-kernel alignment,
quantum-GNN literature -- ground the "bias real, classical still wins"
reconciliation and P4.

---

## PHASE 2: EXPERIMENTAL VALIDATION

**Shared statistical methodology.** Primary unit = per-task Delta-AUC over the 12
Tox21 tasks, paired structured-scrambled, pooled scaffold CV (GroupKFold on Murcko
scaffolds, scaffold-disjoint val+test), averaged over seeds -> 12 paired
observations (as in `run_levelG_probe.py`). Primary test: one-sided **Wilcoxon
signed-rank**; robustness: sign test (`binomtest`). Effect size: median Delta-AUC
+ **Cohen's d** over per-task deltas + **1000-resample bootstrap CI**. Multiplicity:
**Holm-Bonferroni** across the pre-registered (config x K) family; report
Holm-adjusted p (anchor: K=8 raw Wilcoxon p=0.0024 -> Holm-adj 0.017). Power:
MDE at 80% power = **0.0066 AUC** -> size seeds/folds so SE(Delta-AUC) <= 0.002.
For equality-type predictions (P4) use **TOST equivalence testing**, not mere
failure-to-reject. Always report the three negative/null controls: scrambled (null),
meas_only (anti-aligned), separable.

---

### E1: Operator-geometry & information ceiling | Priority: HIGH | Effort: 2-3d | Blocks: [T1, T2, T5]

**Goal.** Verify dim A(O_8) = 3K+2|E| and the DPI ceiling I(X;T_O) <= chi.

**Deliverables**
- [ ] Rank of the HS-Gram matrix of read operators (4^K Pauli-coeff vectors) for K=4,6,8
- [ ] KSG/MINE estimate of I(X;T_O) vs log|X|
- [ ] Holevo chi = S(sum_p rho) - sum_p S(rho) on a molecule sample

**Pass/Fail.** PASS if rank = 3K+2|E| exactly, I(X;T_O) <= chi, and accessible
info scales ~K not 2^K. FAIL (-> revise T1/T2) if MI exceeds chi or grows like 2^K.

**Script changes.**
- [ ] New `probe_operator_geometry.py`
- [ ] Add a `return_state` path to `GraphG.circ` (use `qml.state()`/`qml.density_matrix`)
  to get rho_theta(x)
- [ ] Reuse `featurize` from `run_bias_probe.py`

---

### E2: Gradient-variance / barren-plateau phase transition | Priority: HIGH | Effort: 3-4d | Blocks: [T11(iii), P6]

**Goal.** Confirm poly-K gradient variance for local kappa=2, O(1) depth; and the
2^{-K} collapse at global kappa=K / deep circuits.

**Deliverables**
- [ ] Var_theta[d_theta C] over ~200 random inits, swept over K in {4,6,8,10,12}
- [ ] Depth L in {1,2,4,8}, readout kappa in {1,2,K}
- [ ] log-Var vs K slope analysis

**Pass/Fail.** PASS if log-Var vs K slope ~= -c*log(K) for (kappa=2, L=O(1)) and
~= -K*log2 for kappa=K. FAIL (-> revise Master Thm iii) if the local-cost circuit
also shows 2^{-K} decay.

**Script changes.**
- [ ] New `probe_gradient_variance.py`
- [ ] Expose `n_layers` as an arg in `GraphG` (currently hard-coded 2)
- [ ] Add a `'global'` readout (parity tensor Z_i)
- [ ] Autograd on `pairp`

---

### E3: Spectral low-pass verification | Priority: MED | Effort: 2d | Blocks: [T3]

**Goal.** Confirm coarse features are dominated by low-graph-frequency content
(Lemma 3.5) despite the `discretize` step.

**Deliverables**
- [ ] Energy fraction of cluster-mean features in the bottom-K Laplacian eigenspace
  vs high-frequency band, per molecule
- [ ] Discretization perturbation bound

**Pass/Fail.** PASS if low-freq energy >> high-freq (ideal-low-pass signature) and
discretization perturbation is bounded. FAIL (-> soften T3 to "approximate
low-pass") if cluster-means carry substantial high-frequency content.

**Script changes.**
- [ ] New `probe_spectral_lowpass.py`
- [ ] Expose L=D-A eigendecomposition inside `coarse_graph`
- [ ] Compare Pi_{<=K}f to the cluster-mean `qf`

---

### E4: Phase-A information preservation / epsilon-sufficiency | Priority: MED | Effort: 3-4d | Blocks: [T7 Thm 3.8, T12 eps]

**Goal.** Estimate eps(K) and the substructure-preservation rate; explain the AUC
ceiling 0.61-0.66.

**Deliverables**
- [ ] RDKit substructure-match rate (aromatic rings, functional groups, toxicophores
  intact vs split) for K in {4,6,8}
- [ ] eps ~= AUC(raw-graph GNN) - AUC(coarse model) as I(G;Y)-I(C(G);Y) proxy

**Pass/Fail.** PASS if preservation decreases with fragment size, eps > 0 and larger
at small K, ceiling tracks preservation. FAIL (-> topological bottleneck is vacuous,
revise T7/T12) if coarse ~= raw (eps ~= 0).

**Script changes.**
- [ ] New `probe_info_preservation.py`
- [ ] RDKit `GetSubstructMatches`
- [ ] Raw-graph baseline from `src/models/gnn.py` vs coarse `ClassicalRef`

---

### E5: K-scaling law extension to K=10,12 | Priority: HIGH | Effort: 4-6d (compute-bound) | Blocks: [T10 Prop 3.11, P1]

**Goal.** Extend the linear law and locate the saturation knee K*.

**Deliverables**
- [ ] Median Delta-AUC (levelG struct-scram) at K=10,12 (+re-confirm 4,6,8)
- [ ] Model-selection: linear vs saturating (AIC/BIC or held-out-K CV)

**Pass/Fail.** PASS-P1 if linear (~+0.0014/qubit) through K~=10-12 then bends near
K*~=8-16. Strict linearity to K=16 with no bend -> *refutes* saturation mechanism.
Flat/declining at K=10 -> *refutes* the Theta(K) law (major T10 revision).

**Script changes.**
- [ ] `run_levelG_probe.py --qubits 10 12 --configs levelG gate --seeds 0 1 2
  --out results/kscale.json`
- [ ] Compute optimization: in `GraphG.circ`, when `entangler=='graph'` skip IsingXX
  on pairs with adj[i,j]==0 (identity anyway for sparse graphs)
- [ ] Switch device to `lightning.qubit`
- [ ] Consider bonded-pairs-only correlator readout to cut the O(K^2) observable count
- [ ] Run with `run_in_background`; cache `data/bias_coarse_K10/12.npz`

---

### E6: Alignment-knob alpha interpolation | Priority: MED | Effort: 2-3d | Blocks: [T9 Lemma 4.2, P2]

**Goal.** Trace Delta(lambda) for A_lambda = lambda*A_true + (1-lambda)*A_rand.

**Deliverables**
- [ ] Median Delta-AUC at lambda in {0, 0.25, 0.5, 0.75, 1.0} (placement and
  pooling both use A_lambda)
- [ ] Monotonicity + linearity analysis

**Pass/Fail.** PASS if Delta(lambda) monotone and ~= linear in lambda.
Non-monotone/strongly nonlinear -> refutes the linear place-then-harvest identity
(revise Lemma 4.2).

**Script changes.**
- [ ] In `run_levelG_probe.py`, add `A_interp(AT, AR, lambda)` and a
  `--alpha_lambdas` flag
- [ ] New config that gates+pools with the mixed adjacency

---

### E7: Locality-knob kappa | Priority: MED | Effort: 3-4d | Blocks: [T7 Lemma 3.10, T13, P3]

**Goal.** Delta vs kappa in {1,2,3}: single readout / bond-pooled 2-local /
3-local bond-pooled triples.

**Deliverables**
- [ ] Median Delta-AUC and trainability (grad-var from E2) per kappa
- [ ] Peak identification at toxicophore body-order

**Pass/Fail.** PASS if Delta increases 1->2, and 2->3 iff genuine 3-body
toxicophores (with a trainability cost). Flat 1->2 weakens Lemma 3.10.

**Script changes.**
- [ ] Extend `probe_entanglement_harvesting.py` (already adds IsingZZ + YY)
- [ ] Add a 3-local pooled correlator <O_i O_j O_l> on bonded triples
- [ ] New config `levelG_k3`

---

### E8: Mechanism + anti-alignment | Priority: HIGH | Effort: 1-2d (mostly exists) | Blocks: [T9, phase-diagram anti-aligned]

**Goal.** Re-confirm place-harvest and the anti-aligned region.

**Deliverables**
- [ ] On-bond vs off-bond |C_ij| (target 0.066 vs 0.013 = 5.1x)
- [ ] True-A vs random-A pooled-harvest ratio (target 2.24x vs 0.98x)
- [ ] meas_only Delta (target -0.027, 0/12 tasks positive)
- [ ] Off-bond leakage vs theta^2

**Pass/Fail.** PASS if on/off > 1 with the perturbative hierarchy and meas_only
Delta < 0. meas_only Delta >= 0 -> breaks the anti-aligned prediction.

**Script changes.**
- [ ] Re-run/extend `make_mechanism.py` to K=6,8
- [ ] `run_levelG_probe.py --configs meas_only`

---

### E9: Param-matched equivariant classical GNN -- THE critical experiment | Priority: HIGH | Effort: 3-5d | Blocks: [T12 Thm 4.4, P4]

**Goal.** Test whether the quantum two-qubit-correlator *message* is essential, or
whether a param-matched classical sum-pooled GNN reproduces the struct-scram gap.

**Deliverables**
- [ ] Struct-scram gap for `classicalGNN_pm` (d=7/9/11 ~= 299/435/595 params,
  matching quantum 302/452/610) vs `levelG`
- [ ] Unconstrained MLP absolute AUC baseline
- [ ] Paired (Delta_levelG - Delta_GNN_pm) per task statistic
- [ ] TOST equivalence test (P4 prediction: GNN_pm_struct ~= levelG_struct)

**Pass/Fail.** P4 predicts GNN_pm_struct ~= levelG_struct (both > scrambled, both <
unconstrained MLP) -- test *equivalence* via TOST. If GNN_pm **closes/reproduces**
the gap -> confirms "aggregation is the bias" but **refutes** the non-classical-message
clause 3 of Thm 4.4 -> major theory revision. If GNN_pm shows **no** gap while
levelG does -> confirms the quantum correlator alphabet is essential.

**Script changes.**
- [ ] `run_levelG_probe.py --configs levelG classicalGNN_pm classicalGNN`
  (already wired via CONFIGS)
- [ ] Extend `ClassicalGNN` with a variant whose message is the *plain node-feature
  product* vs one mimicking the two-qubit correlator, to isolate the "alphabet"
  factor
- [ ] Highest publication impact -- run early

---

### E10: Differential feature-injection | Priority: MED | Effort: 3d | Blocks: [T12/P5, T2 tightness]

**Goal.** Test operator-bottleneck tightness: high-graph-frequency (within-cluster)
features should help classical but NOT quantum.

**Deliverables**
- [ ] AUC gain from injecting within-cluster variance/higher-moment features into
  (a) `ClassicalRef` vs (b) `GraphG`
- [ ] Crossed design results

**Pass/Fail.** PASS if classical gains, quantum does not. Quantum gain -> falsifies
bottleneck tightness (revise T2).

**Script changes.**
- [ ] New `probe_feature_injection.py`
- [ ] Augment `coarse_graph` `qf` with within-cluster moments
- [ ] Crossed design over both models

---

### E11: Second-dataset external validity | Priority: MED | Effort: 4-5d | Blocks: [P7]

**Goal.** Test whether the normalized slope (d_K Delta)/d_bar is dataset-invariant
(topology vs idiosyncrasy).

**Deliverables**
- [ ] K-scaling Delta-AUC on BBBP/ClinTox (or a topologically distinct ToxCast
  subset)
- [ ] Normalized slope comparison across datasets

**Pass/Fail.** PASS-P7 if normalized slope invariant across datasets.
Dataset-specific slope -> external-validity blocker remains.

**Script changes.**
- [ ] `run_levelG_probe.py --datasets BBBP ...`
- [ ] Generalize `N_TASKS` handling in `run_bias_probe.py` (BBBP/ClinTox have != 12
  tasks)
- [ ] `featurize` already generalizes from SMILES

---

### E12: Shot & noise robustness as a locality signature | Priority: LOW | Effort: 1-2d (mostly exists) | Blocks: [T11(iii)]

**Goal.** Re-confirm Delta flat in shots and depolarizing p, and overlay the analytic
(1-p)^2 common-rescaling prediction.

**Deliverables**
- [ ] Delta-AUC vs shots in {32,...} (target +0.0184, 100% positive)
- [ ] Delta-AUC vs p in {0,...,0.2} (target +0.0182->+0.0168)
- [ ] (1-p)^2 analytic overlay

**Pass/Fail.** PASS if Delta flat (robust); sharp collapse -> inconsistent with
local-aggregate theory.

**Script changes.**
- [ ] `make_shots.py`, `make_noise.py` (exist); extend to K=8
- [ ] Overlay (1-p)^2

---

### E13: Absorbability bit-exactness audit | Priority: HIGH | Effort: 1-2d (mostly exists) | Blocks: [T8 Thm 2.7/Cor 2.8]

**Goal.** Re-verify L2/L4 scramble is bit-exact absorbable (control vacuous) and
Level 8 gap is non-absorbable under extended training.

**Deliverables**
- [ ] Residual between struct/scram function values for L2/L4 (target 0.00)
- [ ] Level-8 gap stability at --epochs 100

**Pass/Fail.** PASS if L2/L4 residual = 0.00 and Level-8 gap persists. If Level-8
gap -> 0 with more training -> it was absorbable (refutes Cor 2.8(b) application).

**Script changes.**
- [ ] `_verify_absorb.py` (exists)
- [ ] `run_levelG_probe.py --epochs 100`

---

### E14: Representation probe (topology-encoding) | Priority: LOW | Effort: 1d (exists) | Blocks: [T4 / Master Thm ii]

**Goal.** Re-confirm structured features encode graph topology.

**Deliverables**
- [ ] lambda_max(A) linear-probe R^2 from structured vs scrambled features
  (target 0.072 vs 0.040, +80%)
- [ ] Aromaticity tie (expected on non-topological features)

**Pass/Fail.** PASS if structured R^2 > scrambled on a topological target.

**Script changes.**
- [ ] `make_probe.py`, `probe_attention.py` (exist)

---

## CROSS-CUTTING CONCERNS

### What Phase 1 unlocks in Phase 2

- T2 (dimension formula) -> E1's pass/fail target
- T11(iii) (BP threshold) -> E2's slope predictions
- T3 -> E3
- T7 -> E4/E7
- T10 (functional form + K*) -> E5's model-selection
- T9 (overlap identity) -> E6/E8
- T12 (bias-variance + clause 3) -> E9/E10
- Cor 2.8 (pre-registration criterion) -> E13

Wave-0 experiments only re-confirm existing empirics and can run in parallel with
proof-writing; the *interpretation* of E1/E2/E5/E9 requires the corresponding
theorem statement locked first.

### What could falsify the theory and force revision

- **E9**: a param-matched classical GNN reproducing the struct-scram gap -> revise
  Thm 4.4 clause 3 (the "non-classical message" is not necessary) -- the single
  biggest risk to the quantum-specificity story.
- **E5**: flat/declining Delta at K=10 -> revise the Theta(K) scaling law (T10);
  strict linearity to K=16 -> drop the saturation mechanism (P1).
- **E2**: local-cost circuit showing 2^{-K} gradient decay -> revise Master Thm (iii).
- **E1**: measured MI exceeding chi or growing 2^K -> revise T1/T2 (the bottleneck
  framing).
- **E10**: quantum gains from high-frequency features -> operator bottleneck not
  tight (T2).
- **E6**: non-monotone Delta(lambda) -> revise Lemma 4.2.

### Minimum viable theoretical package (for submission)

The defensible core is **T1 (Lemma 1.1) + T2 (Def 1.3, Prop 1.2) + T4 (Thm 2.4)
+ T6 (Thm 2.2, scoped) + T8 (Thm 2.7 + Cor 2.8) + T9 (Lemmas 4.1-4.2)** -- i.e.
*"measurement geometry is a Theta(K)-dimensional, S_K-equivariant, low-complexity,
non-absorbable inductive bias, and it places-then-harvests topology."*

This package proves the bias is real, structured, and methodologically clean without
over-committing to the two riskiest claims. **T10 (scaling law)** and **T11(iii)
(trainability)** are the headline but should be presented as *supported/conditional*
pending **E5** and **E2**; **T12** (why classical wins) rounds out the narrative as
a bias-variance decomposition.

Ship the MVP with Wave-0 + E1 + E9 evidence; add T10/T11 with E5/E2.

---

## PHASE GATE

Before Phase-2 *interpretation* begins, three results must be locked:

1. **T2** -- the exact accessible-subspace dimension formula dim A(O_8) = 3K+2|E|
   (E1's pass/fail depends on it verbatim)
2. **T9** -- the place-then-harvest overlap identity <A',A>/||A'||||A|| (E5/E6/E8
   all reference it)
3. **Cor 2.8** -- the (a)/(b) non-absorbability pre-registration criterion (every
   structured-scrambled comparison must first pass this trace-to-trainable-map check,
   or the control is provably vacuous, as already shown bit-exact for L2/L4)

Wave-0 experiments (E8, E12, E13, E14) may proceed before the gate since they only
re-confirm existing empirics; all other experiments wait on their blocking theorem.

---

## CRITICAL PATH

**Generalization/trainability spine:** T2/T4 -> T6 -> T11

**Scaling/comparative spine:** T2 -> T9 -> T10 -> T12 -> E9

**Backbone (shippable independently):** T8/Cor 2.8

The rate-limiting deliverables are **T6** (the honest generalization bound over the
trained-theta union -- the largest rigor gap) and **T10/E5** (turning a 3-point fit
into a validated law). E9 is the critical *experimental* node; E5 is the critical
*compute* node (start its background run as soon as the E5 script optimization
lands).

---

## GANTT-STYLE EXECUTION ORDER

```
WEEK 1 -- Wave 0 (parallel, immediate -- reuse existing machinery, no theory gate)
  E8   [HIGH]  Re-confirm place-harvest + anti-alignment         1-2d
  E13  [HIGH]  Absorbability bit-exactness audit                 1-2d
  E12  [LOW]   Shot & noise robustness (extend to K=8)           1-2d
  E14  [LOW]   Representation probe (re-run make_probe.py)       1d
  -- Run concurrently with T1, T2, T4, T8 (all root nodes) --

WEEK 1-2 -- Root theorem sprint (no deps, highest priority first)
  T8   [HIGH]  Absorbability theorem (generalize, ship anytime)  2-3d
  T2   [HIGH]  Operator-geometry bottleneck (GATE ITEM)          3-4d
  T4   [HIGH]  Equivariance + GNN identity                       3-4d
  T1   [HIGH]  Quantum MI + Holevo ceiling                       4-6d
  T3   [MED]   Spectral low-pass theory                          4-5d

WEEK 2-3 -- Foundational validation (requires Wave 0 + root theorems)
  E1   [HIGH]  Operator-geometry & info ceiling                  2-3d
  E2   [HIGH]  Gradient-variance / BP phase transition           3-4d
  E3   [MED]   Spectral low-pass verification                    2d
  -- Start E5 background compute run as soon as T2+T9 locked --

WEEK 2-3 -- Downstream theorems (deps on T1/T2/T4)
  T5   [MED]   Q-IB Lagrangian synthesis       deps: T1,T2       2-3d
  T6   [HIGH]  Rademacher bound (CRITICAL PATH) deps: T2,T4      5-7d
  T9   [HIGH]  Place-then-harvest identity (GATE ITEM) deps: T2  3-4d

WEEK 3-4 -- Headline experiments (require gate items T2, T9, Cor 2.8)
  E9   [HIGH]  Param-matched classical GNN (CRITICAL EXPT)       3-5d
  E4   [MED]   Epsilon-sufficiency / info preservation           3-4d

WEEK 3-4 -- Sufficiency chain (deps on T2, T3, T6)
  T7   [HIGH]  Sufficiency chain               deps: T2,T3       4-5d
  T10  [HIGH]  Bias scaling law                deps: T6,T7,T9    4-5d
  T12  [HIGH]  Bias-variance regime theory     deps: T6,T7,T10   4-5d

WEEK 4-5 -- Master theorem + K-scaling results
  T11  [HIGH]  Master Theorem (Thm 4.3)        deps: T5,T6,T7,T9 5-6d
  E5   [HIGH]  K-scaling to K=10,12 (compute-bound)              4-6d

WEEK 5 -- Phase-diagram sweeps
  E6   [MED]   Alpha interpolation sweep                          2-3d
  E7   [MED]   Kappa locality sweep                               3-4d
  E10  [MED]   Differential feature injection                     3d

WEEK 5-6 -- Phase diagram synthesis + external validity
  T12  [HIGH]  (finalize with E9 results)
  T13  [MED]   Kernel-alignment + P1-P8 register deps: T9,T10,T11,T12  3-4d
  E11  [MED]   Second-dataset external validity                   4-5d

PARALLEL ANYTIME -- T8/Cor 2.8 (standalone backbone, ship as soon as done)
```

---

## KEY FILES REFERENCE

| File | Role |
|---|---|
| `run_levelG_probe.py` | Circuit + all Level-8 configs (GraphG, classicalGNN, classicalGNN_pm, meas_only) |
| `run_bias_probe.py` | Featurization, coarse_graph (SpectralClustering), scaffold_folds |
| `probe_entanglement_harvesting.py` | kappa-extension harness (IsingZZ + YY) |
| `src/quantum_levels.py` | 7-level absorbable system |
| `_verify_absorb.py` | Absorbability bit-exactness check (L2/L4 bit-exact) |
| `make_mechanism.py` | Mechanism figures (on/off-bond correlators) |
| `make_shots.py` | Shot-robustness figures |
| `make_noise.py` | Noise-robustness figures |
| `make_probe.py` | Representation probe figures |
| `match_params.py` | Param matching for classicalGNN_pm |
| `run_classical.py` | Classical baselines |
| `src/classical_gnn.py` | ClassicalGNN implementation |
| `src/models/gnn.py` | Raw-graph GNN baseline |

---

## LITERATURE COMPARISON TABLE (Phase I)

This table maps TC-QIC theoretical claims against expectations from the QML and
related literature. Format: Claim | Literature expectation | TC-QIC evidence | Status.

| Claim | Literature expectation | TC-QIC evidence | Status |
|---|---|---|---|
| **Bandwidth / inductive bias** (Canatar et al. 2022) | A scalar bandwidth c rescaling the encoder controls the kernel spectrum decay; without it the fidelity kernel is exponentially flat (eta ~ 2^{-n}) and generalization requires O(3^n) samples. Bias = scalar spectrum-control axis. | TC-QIC GENERALIZES and REPLACES the mechanism. The per-molecule adjacency A = C(mol) acts as an input-conditioned, non-trainable adaptive bandwidth. Critically, a scalar/trainable bandwidth upstream of a variational head is ABSORBABLE (Thm 2.7); TC-QIC works only because A enters as fixed per-molecule data upstream of the only trainable layer (Cor 2.8(b)). The bias is defined information-theoretically as a double bottleneck -- operator geometry (dim A = Theta(K)) + graph low-pass (Pi_{<=K}) -- of which scalar bandwidth is a degenerate special case. | RESOLVED: Canatar provides the alignment motivation; TC-QIC replaces mechanism and adds non-absorbability criterion. |
| **Generalization bounds** (Caro et al. 2022) | Rademacher/covering-number bounds for variational QML scale with the number of gates or trainable parameters. Sample complexity set by circuit depth/parameter count. | TC-QIC Thm 2.2 derives the bound from READOUT DIMENSION Theta(K), not gate count. Bound: R(h) <= R_hat + O(W sqrt(K/n)). Cor 2.3 gives Omega(4^K) -> O(K) sample-complexity saving relative to an unrestricted readout. The union-over-theta encoder term requires an additional covering-number argument (T6 gap); Thm 2.2 currently scoped as conditional on theta plus O(sqrt(dimTheta/n)) encoder term. Caro et al.'s replica-style average-case machinery is complementary and is the natural tool for turning Prop 3.11 into a predicted learning curve. | EXTENDS: readout-locality route is tighter than gate-count route for this architecture; average-case Caro machinery identified as Section D incorporation target. |
| **Barren plateaus** (McClean et al. 2018) | Random deep circuits with global cost observables suffer gradient variance vanishing as 2^{-K}. Cerezo et al. 2021: local cost functions in shallow circuits retain at-most-polynomially-vanishing gradients. | TC-QIC Master Thm clause (iii) invokes the Cerezo local-cost mechanism: bond-pooled readout is a sum of Theta(K) at-most-2-local observables at O(1) depth => gradients stay poly(1/K). The gate is data-dependent (graph-gated IsingXX re-uploading), for which BP theory is less standard; Master Thm (iii) stated as conditional pending E2 empirical verification. Locality is not only a trainability fix but the PRIMARY SOURCE of the operator-geometry inductive bias and shot/noise robustness. | CONDITIONAL: mechanism correctly identified; Cerezo applicability to data-dependent gates unverified -- E2 is the blocking experiment. |
| **Quantum advantage** (Huang et al. 2021; Schuld & Killoran 2022) | Huang et al.: classical data can erase putative quantum advantage; task-model alignment (projected quantum kernel over reduced observables) is decisive, not raw expressivity. Schuld & Killoran: quantum advantage is the wrong goal; characterizing quantum inductive bias is the productive question. | TC-QIC adopts the Schuld-Killoran reframing wholesale: the claim is EXISTENCE AND SCALING OF A TOPOLOGY BIAS, NOT quantum-over-classical. Thm 4.5 (bias-variance decomposition) DERIVES the 5-8 AUC-point classical lead from the same double bottleneck that yields the clean bias: approximation error ||g*_{A_perp}||^2 + epsilon persists while an unconstrained classical MLP sets approximation ~= 0. TC-QIC differs from Huang et al. by conditioning accessible observables on molecular topology PER INPUT rather than defining a single projected kernel. | RESOLVED: reframing adopted; classical dominance derived not asserted; quantum-vs-classical framing explicitly abandoned. |
| **Graph symmetry** (Gilmer et al. 2017; Xu et al. 2019) | MPNN: iterated neighborhood aggregation h_i' = phi(h_i, sum_{j in N(i)} psi(h_i,h_j,A_ij)). Xu et al.: sum-pooling is maximally discriminative among neighborhood aggregators (WL test). Symmetry from weight-tying or gate constraints. | TC-QIC Prop 2.6: the Level-8 readout b[i] = sum_j A_ij C_ij is EXACTLY one MPNN aggregation step differing only in the message alphabet (a genuine two-qubit connected correlator, an entanglement-carrying non-classical quantity). Thm 2.4: S_K-equivariance obtained from MEASUREMENT GEOMETRY ALONE (bond-pooled readout), not weight-tying or gate constraints. Prop 3.3 (orbit sufficiency via Fisher-Neyman): toxicity is S_n-invariant, so the orbit is a sufficient statistic and any non-equivariant model wastes capacity. Lemma 3.5 ties coarse-graining to spectral GNN low-pass filtering (Bruna et al. 2014; Defferrard et al. 2016). | EXTENDS: GNN identity proven; equivariance from measurement geometry is new; spectral low-pass bridge to spectral GNN literature established. |
| **Absorbability** (no direct prior) | Data-encoding determines accessible function/Fourier family (Schuld, Sweke & Meyer 2021). Reparametrization/gauge redundancy in neural losses (Dinh et al. 2017). Re-uploading equivalences (Perez-Salinas et al. 2020). None of these give a function-class-identity theorem for the structured-vs-scrambled control in variational circuits. | TC-QIC Thm 2.7 (NOVEL, NO CLOSE PRIOR): a formal function-class-identity theorem showing the standard structured-vs-scrambled control is VACUOUS under an upstream trainable linear map -- H_struct = H_scram, bit-exact. Already verified bit-exact for L2/L4 in `_verify_absorb.py`. Cor 2.8 gives the exact non-absorbability criterion: (a) >= 2 inconsistent permutations on one shared h, or (b) structure enters as fixed per-molecule data multiplying a physical observable upstream of the only trainable layer. Level 8 satisfies (b). This is the methodological backbone of the entire project: it is the result that makes the Level-8 structured-vs-scrambled comparison scientifically valid and all lower-level comparisons vacuous. | NEW RESULT: strongest novel methodological claim; no analog in kernel, reparametrization, or re-uploading literatures. |

---

## KEY REFERENCES TO CITE (in order of importance)

Listed by decreasing centrality to TC-QIC. Each entry includes author(s), year,
title, venue, and why it matters.

**1.**
  author = {Canatar, Abdulkadir and Bordelon, Blake and Pehlevan, Cengiz},
  year   = {2022},
  title  = {Spectral bias and task-model alignment explain generalization in
            kernel regression and infinitely wide neural networks},
  venue  = {Nature Communications},
  why    = {Primary positioning target. Their bandwidth-c / spectrum-decay /
            alignment framework is the object TC-QIC GENERALIZES: scalar c ->
            per-molecule A; soft spectral decay -> hard rank projection; one
            spectrum -> two (operator geometry + graph Laplacian). Their
            no-generalization (c=1) result is the direct predecessor of Cor 2.3.
            Their replica E_g formula is the prime Section-D incorporation target.}

**2.**
  author = {Schuld, Maria and Killoran, Nathan},
  year   = {2022},
  title  = {Is quantum advantage the right goal for quantum machine learning?},
  venue  = {PRX Quantum},
  why    = {Framing paper whose reframing TC-QIC adopts wholesale: the productive
            question is quantum inductive bias, not quantum-vs-classical. TC-QIC
            provides the concrete falsifiable bias + growth law + non-vacuous
            control that their position paper called for.}

**3.**
  author = {Caro, Matthias C. and others},
  year   = {2022},
  title  = {Generalization in quantum machine learning from few training data},
  venue  = {Nature Communications},
  why    = {The QML generalization bound baseline. Thm 2.2 differs from their
            gate-count-based bound by deriving sample complexity from readout
            dimension Theta(K), yielding a tighter O(W sqrt(K/n)) for this
            architecture. Their average-case / replica machinery is the natural
            tool for turning Prop 3.11 into a predicted learning curve.}

**4.**
  author = {Huang, Hsin-Yuan and Kueng, Richard and Preskill, John},
  year   = {2021},
  title  = {Information-theoretic bounds on quantum advantage in machine learning},
  venue  = {Physical Review Letters},
  why    = {Shows classical data can erase quantum advantage; task-model alignment
            (projected quantum kernel over reduced observables) is decisive. TC-QIC
            extends by conditioning the accessible observables on molecular
            topology per input and by adding the absorbability theorem that their
            fixed-kernel setting cannot see.}

**5.**
  author = {Cerezo, Marco and others},
  year   = {2021},
  title  = {Cost function dependent barren plateaus in shallow parametrized
            quantum circuits},
  venue  = {Nature Communications},
  why    = {Provides the local-cost BP-resistance argument invoked in Master Thm
            clause (iii): at-most-2-local observables + O(1) depth => poly(1/K)
            gradient variance. TC-QIC uses locality not just for trainability but
            as the SOURCE of the operator-geometry inductive bias.}

**6.**
  author = {Gilmer, Justin and Schütt, Kristof T. and others},
  year   = {2017},
  title  = {Neural message passing for quantum chemistry},
  venue  = {ICML},
  why    = {MPNN template that TC-QIC Prop 2.6 instantiates quantumly: the
            bond-pooled two-qubit connected correlator b[i] = sum_j A_ij C_ij is
            exactly one MPNN aggregation step differing only in the message
            alphabet (entanglement-carrying correlator vs learned node-feature
            function).}

**7.**
  author = {Kuebler, Jonas and Buchholz, Simon and Scholkopf, Bernhard},
  year   = {2021},
  title  = {The inductive bias of quantum kernels},
  venue  = {NeurIPS},
  why    = {Argues that quantum measurements supply an inductive bias for molecular
            chemistry. TC-QIC is the direct theoretical development: it identifies
            WHICH measurement (bond-pooled two-qubit connected correlator) yields
            a PROVABLY non-absorbable, K-scaling, topology-aligned bias, and proves
            the positive kernel-alignment increment that Kuebler et al. anticipated.}

**8.**
  author = {McClean, Jarrod R. and Boixo, Sergio and Smelyanskiy, Vadim N. and
            Babbush, Ryan and Neven, Hartmut},
  year   = {2018},
  title  = {Barren plateaus in quantum neural network training landscapes},
  venue  = {Nature Communications},
  why    = {Establishes the global-observable barren-plateau result (gradient
            variance 2^{-K}) that TC-QIC's local-readout design deliberately
            circumvents. The contrast -- local 2-qubit observable vs global
            K-qubit observable -- is the empirical and theoretical motivation for
            the operator-geometry bottleneck.}

**9.**
  author = {Schuld, Maria and Sweke, Ryan and Meyer, Johannes Jakob},
  year   = {2021},
  title  = {Effect of data encoding on the expressive power of variational
            quantum machine learning models},
  venue  = {Physical Review A},
  why    = {Shows data encoding determines the accessible Fourier family / function
            class. TC-QIC Thm 2.7 is the DUAL statement: the READOUT (measurement
            channel) determines the accessible operator subspace, and a trainable
            linear head upstream of that readout can re-absorb any fixed input
            reparametrization -- the absorbability theorem that Schuld et al.'s
            encoding-focused framework cannot see.}

**10.**
  author = {von Luxburg, Ulrike},
  year   = {2007},
  title  = {A tutorial on spectral clustering},
  venue  = {Statistics and Computing},
  why    = {Foundation for Lemma 3.5: the ratio-cut relaxation -> bottom-K
            Laplacian eigenvectors -> ideal graph low-pass projector Pi_{<=K}.
            TC-QIC casts molecular coarse-graining as this ideal low-pass and
            gives "high-frequency atomic noise" a precise definition as the
            discarded high-frequency band, connecting the quantum circuit design
            to the spectral clustering / spectral GNN literature.}
