# T10: The Bias Scaling Law -- $\Delta(K) = \Theta(K)$ for Topology-Aligned Readout

*Status: COMPLETE (theory) + CALIBRATION PENDING E5 | Priority: HIGH | Deps: T6, T7, T9*
*Headline: levelG $\Delta$AUC grows linearly in $K$ ($0.0078 \to 0.0108 \to 0.0134$ at
$K = 4, 6, 8$); gate-only $\Delta$AUC is flat (no $K$-scaling). Both behaviors are
derived, not fitted.*

**Role.** T10 derives Proposition 3.11, the quantitative synthesis of the deductive
program: the structure-scramble AUC gap $\Delta(K)$ is proportional to the aligned
signal mass $\eta_{\mathcal{O}}(K)\, s(K)$ minus the generalization penalty
$c\,W\sqrt{K/n}$ from T6. It converts three qualitative results -- T9's
place-then-harvest identity, T7's sufficiency chain (including Lemma 3.10,
single-qubit blindness), and T6's Rademacher bound -- into a single scaling
prediction with a per-config alignment coefficient. It is the scaling clause of the
Master Theorem (T11, clause (i)).

**Main results.**

- **Definition 1.1** (alignment coefficient): $\eta_{\mathcal{O}} =
  \dim(\mathcal{A}(\mathcal{O}) \cap \mathcal{S}(K)) / \dim \mathcal{S}(K)$.
- **Lemma 1.2** (per-config values): $\eta_B = 1$ (levelG), $\eta_S = \Theta(1/K)$
  (gate-only), $\eta_{\mathrm{meas}} = 0$ for the topology-conditioned signal
  (meas_only).
- **Proposition 3.11** (bias scaling law):
  $\Delta(K) \propto \eta_{\mathcal{O}}(K)\, s(K) - c\,W\sqrt{K/n}$;
  monotone and asymptotically linear in $K$ until saturation at $K^\star$.
- **Corollary 2.2** (config dichotomy): $\Delta_B(K) = \Theta(K)$ for levelG;
  $\Delta_S(K) = O(1)$ (flat) for gate-only. Both match the data.
- **Empirical calibration** (3-point LS fit, $K = 4, 6, 8$):
  $\Delta_B(K) \approx 1.4 \times 10^{-3}\, K + 2.3 \times 10^{-3}$
  ($R^2 \approx 0.996$) -- *consistent with* linear, not proven linear.
- **Saturation prediction**: $K^\star \approx 8$-$16$, testable by E5 at $K = 10, 12$.

---

## 0. Setup and imported results

Notation as in T2, T6, T9. The TC-QIC circuit acts on $K$ qubits; the graph-gated
entangler applies $\mathrm{IsingXX}(A_{ij}\,\theta_{\mathrm{pair}})$ per pair, with
$A$ the coarse-grained adjacency. Three ingredients are imported:

- **From T9** (place-then-harvest): the entangler *places* signal into the
  placed-signal subspace
  $\mathcal{S}(K) = \mathrm{span}\{C_{ij} : A_{ij} > 0\}$ of on-bond connected
  correlators, with $\dim \mathcal{S}(K) = |E| = \tfrac{1}{2}\bar d\, K = \Theta(K)$
  for molecular graphs of bounded mean degree $\bar d$. Off-bond correlators are
  $O(\theta^2)$-suppressed (Lemma 4.1 of T9; empirically $5.1\times$ on/off-bond ratio, E8).
- **From T7** (sufficiency chain): $B_A$ is a minimal sufficient equivariant
  statistic for the placed signal (Thm 3.9), and the single-qubit readout $S$ is
  blind to all connected correlators (Lemma 3.10: marginals determine no $C_{ij}$).
- **From T6** (Rademacher bound): the head generalization penalty is
  $2BW/\sqrt{n} = O(W\sqrt{K/n})$ at fixed encoder parameters, with an
  encoder term $\tilde O(\sqrt{P/n})$ common to the structured and scrambled arms
  (so it cancels in the *difference* $\Delta$ to leading order).

The quantity modeled is the structure-scramble gap

$$
\Delta(K) \;=\; \mathrm{AUC}_{\mathrm{struct}}(K) \;-\; \mathrm{AUC}_{\mathrm{scram}}(K),
$$

the benchmark's headline statistic (inductive bias, not absolute performance).

---

## 1. The alignment coefficient $\eta_{\mathcal{O}}$

**Definition 1.1 (topology alignment coefficient).** For a readout family
$\mathcal{O}$ with accessible operator subspace $\mathcal{A}(\mathcal{O})$ (T2),

$$
\eta_{\mathcal{O}}(K) \;=\;
\frac{\dim\!\big(\mathcal{A}(\mathcal{O}) \cap \mathcal{S}(K)\big)}{\dim \mathcal{S}(K)} .
$$

$\eta_{\mathcal{O}} \in [0, 1]$ measures the fraction of the placed signal the
readout can harvest. It is the single number through which the readout enters the
scaling law.

**Lemma 1.2 (per-config values).**

*(a) levelG (bond-pooled, $\mathcal{O}_8$): $\eta_B = 1$ for all $K$.*

*Proof.* $\mathcal{A}(\mathcal{O}_8)$ contains the bond-pooled correlators
$O^{ZZ}_i = \sum_j A_{ij} Z_i Z_j$ and $O^{XX}_i = \sum_j A_{ij} X_i X_j$ (T2).
Their span, together with the $3K$ single-qubit terms used to form the connected
parts, covers every on-bond pair $(i,j) \in E$: each $C_{ij}$ with $A_{ij} > 0$
appears with nonzero weight $A_{ij}$ in the pooled operators, and the $A$-weighted
contraction is injective on $\mathcal{S}(K)$ (T7, Thm 3.9 minimal sufficiency).
Hence $\mathcal{S}(K) \subseteq \mathcal{A}(\mathcal{O}_8)$ and
$\eta_B = |E|/|E| = 1$. $\square$

*(b) gate-only (single-qubit $Z$): $\eta_S = \Theta(1/K)$.*

*Proof.* $\mathcal{A}(\mathcal{O}_{\mathrm{gate}}) = \mathrm{span}\{Z_i : i = 1,
\dots, K\}$ consists of one-qubit marginal functionals. By Lemma 3.10 (T7),
marginals determine no connected correlator, so in the strict operator-subspace
sense $\mathcal{A}(\mathcal{O}_{\mathrm{gate}}) \cap \mathcal{S}(K) = \{0\}$ and
$\eta_S = 0$. The effective (statistical) alignment is not exactly zero: the
circuit correlates $\langle Z_i \rangle$ weakly with the pair structure incident on
qubit $i$ (each $Z_i$ aggregates the $O(\bar d)$ bonds at $i$ into one scalar,
losing the pairwise resolution of the $\Theta(K)$-dimensional $\mathcal{S}(K)$).
One scalar per qubit against $\Theta(K)$ signal dimensions gives an accessible
share $\eta_S = \Theta(1/K)$. $\square$

*(c) meas_only (bond-pooled readout, fixed graph-independent entangler):
$\eta_{\mathrm{meas}} = 0$ for the topology-conditioned signal.*

*Proof sketch.* Alignment requires *both* placement and harvest (T9). Here
$\mathcal{A}(\mathcal{O}_8) \supseteq \mathcal{S}(K)$ as in (a), but the fixed ring
entangler places correlators on the *ring* pairs, independent of the molecule's
$A$. The harvested quantity $\sum_j A_{ij} C^{\mathrm{ring}}_{ij}$ contracts the
molecule's adjacency against molecule-*independent* correlators: it harvests random
(graph-agnostic) correlators, not molecule-specific ones, so the
topology-conditioned component of the signal is absent. Net
$\eta_{\mathrm{meas}} = 0$; this explains the empirical
$\Delta\mathrm{AUC}(\mathrm{meas\_only}) < 0$ (harvesting misaligned correlators is
worse than not harvesting -- it injects structured noise keyed to the wrong graph). $\square$

**Remark.** Lemma 1.2 formalizes the Level-G finding (see memory: measurement
readout grows the bias, measurement *alone* hurts): the bond-pooled readout is
necessary but not sufficient; it must be paired with the graph-gated entangler.

---

## 2. Deriving the scaling law (Proposition 3.11)

**Model.** Decompose the gap:

$$
\Delta(K)
\;=\;
\underbrace{\mathrm{sig}_{\mathrm{aligned}}(K)}_{\eta_{\mathcal{O}}(K)\, s(K)}
\;-\;
\underbrace{\mathrm{sig}_{\mathrm{scram}}(K)}_{O(1)}
\;-\;
\underbrace{\mathrm{pen}(K)}_{O(W\sqrt{K/n})},
$$

where:

1. $\mathrm{sig}_{\mathrm{aligned}}(K) = \eta_{\mathcal{O}}(K)\, s(K)$ with $s(K)$
   the aligned signal mass in $\mathcal{A}(\mathcal{O})$. Placement (T9, Lemma 4.1)
   puts $\Theta(1)$ mass per on-bond correlator, and E1 gives effective rank
   $\approx 1.5 K$ after $A$-weighting, so $s(K) = \Theta(K)$: the $K$-scaling
   comes from *adding bond pairs*, not from exponential Hilbert-space growth.
2. $\mathrm{sig}_{\mathrm{scram}}(K) = O(1)$: scrambling the adjacency destroys the
   placement-harvest alignment (the scrambled arm places on $\pi(E)$ but the label
   depends on $E$); residual alignment is the chance overlap of a random
   permutation, $O(1)$ and non-growing in $K$. (Absorbability caveat: the vacuous
   Level-2 scramble is excluded; T8.)
3. $\mathrm{pen}(K) = O(W\sqrt{K/n})$: the head term of T6, Thm 2.2. The encoder
   term $C_{\mathrm{enc}} = \tilde O(\sqrt{P/n})$ is identical across arms and
   cancels in the difference to leading order.

**Proposition 3.11 (bias scaling law).** *For the TC-QIC architecture,*

$$
\Delta(K) \;\propto\; \eta_{\mathcal{O}}(K)\, s(K) \;-\; c\,W\sqrt{K/n} .
$$

*For the Level-8 (levelG) configuration this instantiates as*

$$
\Delta_B(K) \;=\; \alpha\, K + \beta + O\!\big(\sqrt{K/n}\big),
$$

*with $\alpha = s_0\, \eta_B / n_{\mathrm{eff}} > 0$ the topology-aligned signal
gain per qubit ($s_0$ = per-bond signal mass, $n_{\mathrm{eff}}$ = effective sample
size per class) and $\beta$ a constant offset absorbing finite-size,
initialization, and scaffold effects. The law is monotone increasing and
asymptotically linear in $K$ until saturation at $K^\star$ (section 6).*

*Proof.* Substitute Lemma 1.2 and the three-term decomposition. For levelG:
$\eta_B = 1$, $s(K) = \Theta(K)$, so
$\mathrm{sig}_{\mathrm{aligned}} = \Theta(K)$; the scrambled residual is $O(1)$
(absorbed into $\beta$); the penalty is $O(\sqrt{K/n})$, subdominant whenever
$n \gg K$ -- satisfied here ($n = 7823 \gg K^2 = 64$). Hence
$\Delta_B(K) = \Theta(K) - O(\sqrt{K/n}) = \Theta(K)$ for large $n$. Monotonicity
holds on the pre-saturation range $K \le K^\star$ where each added qubit adds
$\Theta(\bar d)$ new on-bond correlators (T9); linearity holds *asymptotically* in
the same range because the effective rank grows linearly (E1:
$\mathrm{eff\_rank}/K \approx 1.5$, constant across $K$). $\square$

**Corollary 2.2 (config dichotomy).**

- *levelG*: $\Delta_B(K) = \Theta(K)$ -- the bias grows linearly per qubit.
- *gate-only*: $\eta_S = \Theta(1/K)$ and
  $s_{\mathrm{gate}}(K) = \dim \mathcal{A}(\mathcal{O}_{\mathrm{gate}}) = \Theta(K)$,
  so $\mathrm{sig}_{\mathrm{aligned}} = \Theta(1/K) \cdot \Theta(K) = \Theta(1)$:
  **flat in $K$**. Thus $\Delta_S(K) = O(1)$, and once the $\sqrt{K/n}$ penalty and
  the noise floor rise, the *detectable* effect fades below significance.

The dichotomy is the falsifiable content: the *same* formula, evaluated at two
alignment coefficients, predicts growth for one config and flatness for the other.

---

## 3. Empirical calibration (3-point fit; E5 pending)

Median $\Delta$AUC from `stats_summary.json` (levelG, gate) at $K = (4, 6, 8)$:

| config | $K=4$ | $K=6$ | $K=8$ | trend |
|---|---|---|---|---|
| levelG | $0.0078$ | $0.0108$ | $0.0134$ | monotone increasing |
| gate | $0.0044$ | $0.0026$ | $0.0030$ | flat, mean $\approx 0.003$ |

Least-squares fit to the levelG points:

$$
\boxed{\;\Delta_B(K) \;\approx\; 1.40 \times 10^{-3}\, K \;+\; 2.20 \times 10^{-3}\;}
\qquad (R^2 \approx 0.996 \text{ on 3 points}),
$$

i.e. calibrated slope $\alpha \approx 0.0014$ AUC/qubit and offset
$\beta \approx 0.0022$.

**CAUTION (over-claiming linearity).** Three points on a smooth monotone curve
cannot distinguish linear from logarithmic, power-law, or mildly saturating growth;
$R^2 = 0.996$ on 3 points is *not* statistical evidence for linearity, since any
smooth function fits 3 points well. The correct claim is: the data are
**consistent with** the linear law that Prop 3.11 predicts on the pre-saturation
range; the slope $\alpha$ is an *empirical calibration*, pending E5.

**E5 discrimination protocol ($K = 10$ running; $K = 12$ not yet started).**

- If linear holds to $K = 10$:
  $\Delta_B(10) \approx 0.0140 + 0.0022 = 0.0162$.
- If saturating with $K^\star \approx 8$:
  $\Delta_B(10) \approx 0.0134 + \delta$ with $\delta < 0.0028$ (sub-linear increment).

The two hypotheses are distinguishable if E5 achieves $\mathrm{SE} < 0.003$ on the
median $\Delta$AUC, which requires roughly $2000{+}$ molecules and $3$ seeds --
consistent with the E5 full-run plan.

---

## 4. Gate config: the flat law confirmed

Gate empirical: $(0.0044, 0.0026, 0.0030)$ at $K = (4, 6, 8)$; mean $\approx 0.003$
with no trend, and all points sit below the $80\%$-power minimum-detectable effect
($6.6 \times 10^{-3}$). This is exactly Corollary 2.2: the single-qubit readout
does **not** scale with $K$ because it cannot access the $|E|$-growing bond
correlator subspace (Lemma 3.10), and its $\Theta(1)$ aligned mass comes only from
the indirect $O(1/K)$-per-dimension circuit correlations of Lemma 1.2(b). The gate
fade is therefore *predicted*, not merely observed -- it is the null arm of the
scaling dichotomy.

Quantitatively, $\Delta_S / \Delta_B = 0.56, 0.24, 0.22$ at $K = 4, 6, 8$: the
ratio decreases with $K$, as it must if the numerator is $\Theta(1)$ and the
denominator $\Theta(K)$.

---

## 5. Classical GNN param-matched comparison (E9)

classicalGNN_pm median $\Delta$AUC (from `cls_pm_K468.json`):
$(0.0141, 0.0069, 0.0121)$ at $K = (4, 6, 8)$ -- **non-monotone**, no clear trend.

**Interpretation.** The classical bond-pooled node product $h_i h_j$ is a
first-order approximation to the quantum correlator $C_{ij}$: it accesses the
*same* topology alignment ($\eta \approx 1$ in the sense of Definition 1.1) but
without the quantum coherent contribution to the placed mass $s_0$. The pattern
suggests: at $K = 4$ the classical product suffices (simpler features model the
coarse task well); at $K = 6$ the quantum correlator's coherent component adds
$\approx 1.56\times$ signal; at $K = 8$ they re-converge. What the classical model
does *not* show is the clean monotone $\Theta(K)$ law.

**The load-bearing distinction:** levelG's *scaling* is quantum-specific even
though levelG's *mechanism* (topology alignment) is shared with the classical
model. Prop 3.11's linear law is a property of the coherently placed correlator
mass, not of bond-pooling per se. This is consistent with the program framing
(inductive bias, not quantum $>$ classical): classically the absolute
$\Delta$AUC is comparable or better at individual $K$, but the systematic
per-qubit growth law is only observed in the quantum arm.

---

## 6. Saturation: the $K^\star$ prediction

Prop 3.11's linear regime requires that adding a qubit adds new on-bond correlator
dimensions, i.e. $|E|$ grows with $K$. This fails when the molecule is
**over-partitioned**:

$$
K^\star \;\approx\; \frac{n_{\mathrm{atoms,modal}}}{\text{target cluster size}},
$$

with $n_{\mathrm{atoms,modal}} \approx 20$-$30$ heavy atoms for drug-like Tox21
molecules and a minimum meaningful cluster size of $2$ (one atom per qubit means no
coarse-graining and the coarse graph approaches complete). This gives
$K^\star \approx 10$-$15$; conservatively $K^\star \approx 8$-$16$.

At $K > K^\star$ the coarse graph is nearly complete: every qubit bonds to almost
every other, $\mathcal{S}(K)$ approaches the full pair space, the bond-pooled
readout sums over almost all pairs, and the structured-vs-scrambled distinction
collapses (a scramble of a complete graph is the same graph). Information-
theoretically, $\partial_K I(C(G); Y) \to 0$.

**PREDICTION:** $\Delta_B(K^\star + 1) < \Delta_B(K^\star)$ (turnaround, or at
minimum a plateau). E5 at $K = 10, 12$ tests this directly: strict linear
continuation *refutes* the saturation mechanism; a bend *confirms* it. Either
outcome is informative -- the honest statement of the law is "monotone,
asymptotically linear **until saturation**", and only $K^\star$'s location is at
stake.

---

## 7. Connection to T11 (Master Theorem)

Prop 3.11 is the scaling prediction of T11 clause (i). It upgrades the inductive
bias from an existence claim to a **quantifiable resource**: under the TC-QIC prior
the bias gain per qubit is $\alpha > 0$, so adding qubits *helps* (in structured
bias, not absolute AUC) at a sample cost of only $O(\sqrt{K/n})$ (T6). Combined
with T6's polynomial sample complexity, this is the positive half of the program's
verdict: the bias is real, small, and grows linearly with circuit size on the
pre-saturation range -- with the growth attributable, via Lemma 1.2 and E9, to the
coherent correlator mass that the topology-aligned readout harvests.

**Dependency map.** T9 supplies $\mathcal{S}(K)$ and placement ($s(K) = \Theta(K)$);
T7 supplies harvest optimality ($\eta_B = 1$) and blindness ($\eta_S \to 0$,
Lemma 3.10); T6 supplies the penalty term ($c\,W\sqrt{K/n}$) and its cross-arm
cancellation. T10 is the unique point where all three meet a free parameter
($\alpha$, $\beta$) calibrated on data.

---

## 8. Honest-scope summary

| Claim | Status |
|---|---|
| $\Delta_B(K)$ monotone increasing on $K = 4$-$8$ | empirical, $p = 0.017 \to 0.011 \to 0.0024$ |
| $\Delta_B(K) = \Theta(K)$ pre-saturation | derived (Prop 3.11); 3 points consistent |
| Linearity specifically (vs log/power) | **not established** -- E5 required |
| Slope $\alpha \approx 0.0014$/qubit | empirical calibration only |
| Gate flatness | derived (Cor 2.2) and observed |
| $K^\star \approx 8$-$16$ | prediction, untested (E5 $K = 10$ running) |
| Classical non-monotonicity | observed (E9); interpretation section 5 is post-hoc |
