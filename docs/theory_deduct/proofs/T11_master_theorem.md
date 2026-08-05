# T11: The Master Theorem -- TC-QIC Provides a Structural, Topology-Conditioned Prior

*Status: COMPLETE | Priority: HIGH | Deps: T5, T6, T7, T9 (all done)*
*This is the headline synthesis theorem of the TC-QIC framework.*

**Role.** T11 is the capstone of the deductive program. It assembles the three
main TC-QIC pillars -- automatic topology-aligned compression (T2, T3, T5, T9),
$S_K$-equivariance with polynomial generalization (T4, T6), and conditional
local-cost trainability (T2, T9, Cerezo 2021 + E2) -- into a single structural
statement. T11 does NOT introduce new machinery; it certifies that the completed
lemmas fit together and it fixes the precise scope of each clause.

---

## 0. Preamble: what the Master Theorem does NOT claim

Before the statement, four common misreadings that the phrasing must explicitly
avoid. These are load-bearing scope conditions, not disclaimers.

1. **Not "quantum $>$ classical" in absolute AUC.** E9 shows the parameter-matched
   classical model `classicalGNN_pm` achieves a comparable or larger
   struct-scram gap (Corollary 4.3b). The claim of T11 is about a STRUCTURED,
   topology-aligned BIAS, not about overall predictive performance.

2. **Not "the unitary is forbidden from Hilbert space."** The graph-gated unitary
   $U_\theta$ still explores the full $2^K$-dimensional Hilbert space. It is the
   MEASUREMENT channel -- the fixed Level-8 readout $\mathcal{O}_8$ -- that
   projects onto a $\Theta(K)$-dimensional operator slice. The bottleneck lives in
   the observable algebra $\mathcal{A}(\mathcal{O}_8)$, not in the state space.

3. **Clause (iii) (trainability) is NOT unconditionally proven.** It requires the
   circuit to be shallow ($O(1)$ depth) and the cost to be 2-local. These hold for
   Level-8. However, the formal barren-plateau guarantee of Cerezo et al. (2021)
   requires block-local random circuits with local 2-design structure, which the
   data-dependent re-uploading GraphG circuit does NOT possess. Clause (iii) is
   therefore SUPPORTED CONDITIONALLY and verified empirically at $K = 4,6,8$ by E2.

4. **The generalization bound (T6) is SCOPED.** It gives the worst-case DOMINANT
   term $O(W\sqrt{K/n})$ for the linear head at fixed encoder parameters, but the
   full bound over the trained encoder includes an additive encoder-complexity
   term $C_{\mathrm{enc}} = \tilde O(\sqrt{P/n})$ with $P = O(K^2)$ that must NOT
   be suppressed. The exponential saving is a statement about the DOMINANT
   head-driven term only.

---

## 1. Setup

Let $G$ be a molecular graph and $C(G)$ its $K$-cluster spectral coarsening
(T3). Let $\rho_\theta(C(G)) \in \mathcal{D}(\mathcal{H})$,
$\mathcal{H} = (\mathbb{C}^2)^{\otimes K}$, be the $K$-qubit state prepared by the
GraphG circuit with parameters $\theta$: an RY re-uploading encoder followed by
the graph-gated entangler $\mathrm{IsingXX}(A_{ij}\,\theta_{\mathrm{pair}})$,
where $A$ is the max-normalized adjacency of $C(G)$ ($\max_{ij} A_{ij} \le 1$).

The Level-8 bond-pooled readout is the affine feature map

$$
\phi_{\mathcal{O}_8}(\rho_\theta)
= \big[\, \langle X_i\rangle,\ \langle Y_i\rangle,\ \langle Z_i\rangle
   \ ;\ B_A^{ZZ},\ B_A^{XX} \,\big] \in \mathbb{R}^{5K},
$$

with $3K$ single-qubit Pauli expectations and $2K$ bond-pooled correlators

$$
\big(B_A^{ZZ}\big)_i = \sum_j A_{ij}\, \langle Z_i Z_j\rangle,
\qquad
\big(B_A^{XX}\big)_i = \sum_j A_{ij}\, \langle X_i X_j\rangle .
$$

The classification head is the linear map $h_w(\phi) = w^\top \phi$ with
$\|w\|_2 \le W$. Let $C_{ij}(\rho) = \langle Z_i Z_j\rangle - \langle Z_i\rangle
\langle Z_j\rangle$ denote the connected correlator (T9).

---

## THEOREM 4.3 (Master Theorem): The TC-QIC Structural Prior

The TC-QIC architecture $(C, U_\theta, \mathcal{O}_8, h_w)$ satisfies the
following THREE properties simultaneously.

### (i) Topology-aligned tight compression

The readout $\phi_{\mathcal{O}_8}$ is a $\Theta(K)$-dimensional,
$S_K$-equivariant projection of the molecular graph structure. Specifically:

- **(a) [T2] Operator-geometry bottleneck.**
  $\dim \mathcal{A}(\mathcal{O}_8) = 3K + 2|E| = \Theta(K) \ll 4^K$,
  where $|E|$ is the number of bonded pairs of $C(G)$. The accessible observable
  algebra is a $\Theta(K)$-dimensional slice of the $4^K$-dimensional full Pauli
  space.

- **(b) [T3] Spectral bottleneck.** The coarse graph
  $C(G) = \Pi_K G + \varepsilon_{\mathrm{disc}}$, where $\Pi_K$ is the bottom-$K$
  Laplacian projector, retains $\Theta(K)$-many low-frequency Laplacian modes and
  discards high-frequency atomic noise. Empirically (E3) $93\%$ of the spectral
  energy lies in the bottom $K/2$ modes, with discretization loss
  $\varepsilon_{\mathrm{spectral}} < 7\%$.

- **(c) [T9] Place-then-harvest identity.** The graph-gated entangler PLACES
  signal proportionally to $A$ (Lemma 4.1) and the bond-pooled readout HARVESTS
  exactly the placed signal (Lemma 4.2):

  $$
  \big(B_A(\rho)\big)_i = \sum_j A_{ij}\, C_{ij}(\rho),
  $$

  with the off-bond subspace $S_\perp$ ($A_{ij}=0$) receiving no gate and
  contributing nothing to the readout (E8: $5.1\times$ on-bond vs off-bond
  correlator ratio).

**Consequence (information ladder).** The information reaching the head is
bounded by a monotone chain:

$$
I(G;Y)\ \ge\ I(C(G);Y)\ \ge\ I\big(\phi_{\mathcal{O}_8}(\rho_\theta);\,Y\big),
$$

the first inequality by T3 / T7 (with $\varepsilon_{\mathrm{spectral}} < 7\%$,
E3), the second by the data-processing inequality through the
$\Theta(K)$-dimensional readout slice (T2). The compression is AUTOMATIC (no
explicit regularizer -- it is baked into the fixed operator geometry) and
TOPOLOGY-ALIGNED (the retained coordinates are $A$-weighted).

### (ii) $S_K$-equivariance and polynomial generalization

- **(a) [T4] Equivariance.** For any qubit permutation $\pi \in S_K$ realized by
  the permutation unitary $U_\pi$,

  $$
  \phi_{\mathcal{O}_8}\big(U_\pi \rho\, U_\pi^\dagger\big)
  = \pi \cdot \phi_{\mathcal{O}_8}(\rho),
  $$

  i.e. relabeling the qubits permutes the output coordinates (and the
  corresponding rows/columns of $A$) without changing their values. The
  bond-pooled readout is exactly permutation-equivariant.

- **(b) [T6, scoped] Generalization.** For the linear head at fixed encoder
  parameters, with probability $\ge 1-\delta$ over $n$ i.i.d. samples,

  $$
  R(h) - \hat R_n(h)
  \ \le\ \frac{2 B W}{\sqrt n}
  \ +\ C_{\mathrm{enc}}(\Theta, n, \delta)
  \ +\ 3\sqrt{\tfrac{\ln(2/\delta)}{2n}},
  $$

  where $B = \sup_\rho \|\phi_{\mathcal{O}_8}(\rho)\|_2 \le
  \sqrt{3K + 2K\bar d^{\,2}} = \Theta(\sqrt K)$ (T6 Lemma 2.1). The DOMINANT
  head-driven term is therefore $O(W\sqrt{K/n})$. The encoder term
  $C_{\mathrm{enc}} = \tilde O(\sqrt{P/n})$ with $P = O(K^2)$ encoder parameters
  (proved self-contained by a Lipschitz-in-$\theta$ covering argument in T6, and
  cross-checked against Caro et al. 2022 and Abbas et al. 2021) is subdominant
  once $n \gg K^2$.

  **Exponential saving (dominant term).** Learning to accuracy $\varepsilon$
  requires $n = \Omega(K)$ samples for the $5K$-feature readout, versus
  $n = \Omega(4^K)$ for the full-Pauli readout ($B_{\mathrm{full}} = 2^{K/2}$
  exactly on pure states). This saving is a statement about the dominant term
  ONLY; the full bound retains $C_{\mathrm{enc}}$, which is common to the
  structured and scrambled arms of the benchmark.

- **(c) [T5] Auto-compression.** The Q-IB Lagrangian is automatically in the
  strong-compression regime:

  $$
  I\big(X;\, T_{\mathcal{O}_8}\big) = O(K \log K)
  \quad (\le K \text{ bits via the T1 Holevo ceiling}),
  $$

  independent of $\theta$ and of the tradeoff $\beta$ -- because the compression
  term is capped by the operator geometry (T2) and the bounded feature cube
  before any optimization.

### (iii) Local-cost trainability (conditional)

The cost $C(\theta) = \mathbb{E}\big[\ell\big(w^\top
\phi_{\mathcal{O}_8}(\rho_\theta),\, Y\big)\big]$ is a WEIGHTED SUM of
at-most-2-local observables: every readout term $C_{ij} = \langle Z_i Z_j\rangle$
or $\langle X_i X_j\rangle$ acts nontrivially on at most two qubits (T2, T9).
Cerezo et al. (2021, Theorem 1) then guarantee, for shallow ($O(1)$-depth)
circuits with 2-local cost observables,

$$
\mathrm{Var}_\theta\!\left[\frac{\partial C}{\partial \theta_l}\right]
\ \ge\ \frac{1}{\mathrm{poly}(K)}
\qquad (\text{not } 2^{-K}),
$$

i.e. no exponentially vanishing gradients.

**Honest scope.** The Cerezo theorem is proved for block-local RANDOM circuits
with local 2-design structure. The Level-8 data-re-uploading GraphG circuit is
NOT a random circuit -- its parameters are data-dependent and its entangler is
graph-gated -- so the theorem's hypotheses are not literally met. Applicability
is therefore CONDITIONAL on the empirical gradient variance remaining
polynomial-in-$K$.

**E2 status.** Circuit-level $\theta$-parameters show flat variance-times-$K$
products $\mathrm{Var}\cdot K = 2.8\times10^{-4} / 6.8\times10^{-4} /
6.6\times10^{-4}$ at $K = 4,6,8$ -- consistent with $O(1)$, i.e. no exponential
decay. The $K$-range is too narrow for a definitive conclusion; E5 ($K=10$) will
provide indirect evidence (a trainable circuit implies the bias should continue
to scale with $K$).

**Clause (iii) verdict.** SUPPORTED CONDITIONALLY; verified empirically at
$K = 4$--$8$ by E2. It is NOT claimed as unconditionally proven.

---

## COROLLARY 4.3a: Why bond-pooling beats single-qubit readout

The `gate` configuration (single-qubit $Z$ readout only) has
$\dim = 3K$, versus $3K + 2|E|$ for `levelG`. The extra $2|E|$ bond-correlator
dimensions are BLIND to single-qubit marginals: no linear functional of
$\{\langle Z_i\rangle\}$ recovers $\langle Z_i Z_j\rangle - \langle Z_i\rangle
\langle Z_j\rangle$ (Lemma 3.10). Empirically the struct-scram gap ratio

$$
\frac{\Delta\mathrm{AUC}(\mathrm{gate})}{\Delta\mathrm{AUC}(\mathrm{levelG})}
= 0.44 / 0.27 / 0.22 \quad\text{at } K = 4/6/8,
$$

DECREASES with $K$: because $|E|/K = (K-1)/2$ grows, the bond-pooled advantage
compounds as the graph grows. $\square$

---

## COROLLARY 4.3b: Why classical bond-pooling also shows the gap (E9)

The parameter-matched classical model `classicalGNN_pm` (bond-pooled products of
node features $h_i h_j$) ALSO shows a statistically significant struct-scram gap
(Wilcoxon $p = 0.002 / 0.026 / 0.0005$ at $K = 4/6/8$).

**Interpretation.** TC-QIC correctly PREDICTS this. The gap originates from the
ALIGNMENT between the readout topology $A$ and the true molecular topology
(clause (i)), NOT from quantum coherence per se. Any $A$-weighted bond-pooled
readout -- quantum or classical -- accesses that alignment. The quantum
correlator $C_{ij} = \langle Z_i Z_j\rangle - \langle Z_i\rangle\langle
Z_j\rangle$ adds entanglement-sensitive information ON TOP of the classical
product $h_i h_j$; this additive quantum contribution is real but
regime-dependent (at $K=6$, $1.56\times$ larger median $\Delta\mathrm{AUC}$).

**Revision to Thm 4.4 clause (ii).** The "non-classical message" is a
REGIME-DEPENDENT ADDITIVE contribution, not a necessary condition for the gap.
$\square$

---

## PROOF SKETCH (assembling T5 / T6 / T7 / T9)

### (i) Topology-aligned compression

Chain the four completed lemmas along the data path
$G \to C(G) \to \rho_\theta \to \phi_{\mathcal{O}_8}(\rho_\theta) \to$ head:

- **T3** gives $C(G) = \Pi_K G + \varepsilon_{\mathrm{disc}}$: coarsening is the
  bottom-$K$ Laplacian projection, losing at most $\varepsilon_{\mathrm{spectral}}$.
- **T9 Lemma 4.1** (placement): a single $\mathrm{IsingXX}(\theta)$ places
  $C_{ij}(\rho) = -\sin^2(\theta)(1-\langle Z_i\rangle_0^2)(1-\langle Z_j\rangle_0^2)$,
  i.e. correlation proportional to $A_{ij}$ to leading order in $\theta$.
- **T9 Lemma 4.2** (harvest): $B_A$ harvests exactly the placed-signal subspace
  $S(K)$; the off-bond subspace $S_\perp$ is blind ($5.1\times$ empirical ratio).
- **T2** certifies $\phi_{\mathcal{O}_8}$ is a projection onto a
  $3K + 2|E|$-dimensional subspace of the $4^K$-dimensional operator space.

Each arrow loses at most $\varepsilon_{\mathrm{spectral}}$ (T3),
$\varepsilon_{\mathrm{operator}}$ (T2), $\varepsilon_{\mathrm{readout}}$ (T9);
composing gives the information ladder and the "automatic + topology-aligned"
conclusion.

### (ii) Equivariance + generalization

- **T4** gives $S_K$-equivariance of $B_A$ by the conjugation identity
  $\phi_{\mathcal{O}_8}(U_\pi \rho U_\pi^\dagger) = \pi\cdot\phi_{\mathcal{O}_8}(\rho)$.
- **T6** gives the Bartlett-Mendelson bound with the bounded-feature Lemma 2.1
  ($B = \Theta(\sqrt K)$) for the dominant head term, plus the Caro-style
  covering term $C_{\mathrm{enc}}$ for the encoder.
- **T5** gives the auto-compression $I(X;T_{\mathcal{O}_8}) = O(K\log K)$ from
  T1 + T2.

### (iii) Trainability

T9 proves the cost is a sum of 2-local terms; Cerezo (2021) applies CONDITIONALLY
(2-design hypotheses not literally met), with E2 supplying the empirical
polynomial-variance evidence at $K = 4$--$8$. $\square$

---

## REMARK: Scope vs the original "double bottleneck prevents barren plateaus" claim

The original Q-TIB formulation (`action_plan.md`) claimed the "double bottleneck
prevents barren plateaus." TC-QIC is more precise, and separates two DISTINCT
mechanisms:

- **Readout locality** (bond-pooled 2-local observables) yields
  polynomial-in-$K$ gradient variance -- clause (iii), conditional on Cerezo +
  E2.
- **Spectral bottleneck** (T3) prevents over-smoothing by discarding
  high-frequency noise -- following from T3 + T7, unconditional.

The "prevents barren plateaus" claim is thus conditional on clause (iii); the
"prevents over-smoothing" claim follows from T3 + T7 directly. Conflating the two
was the imprecision T11 removes.

---

*Dependencies discharged: T2 (operator geometry), T3 (spectral low-pass),
T4 (equivariance), T5 (Q-IB Lagrangian), T6 (Rademacher, scoped), T7 (sufficiency
chain), T9 (place-then-harvest). Empirical support: E2, E3, E8, E9. Conditional
items flagged inline: clause (iii) trainability and the encoder term
$C_{\mathrm{enc}}$ in clause (ii).*
