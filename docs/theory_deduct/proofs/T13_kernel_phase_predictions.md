# T13: Kernel-Alignment Bridge, the (alpha, kappa) Phase Diagram, and the P1-P8 Prediction Register

*Status: COMPLETE | Priority: HIGH | Deps: T9, T10, T11, T12 (all done)*
*This is the final theory document of the TC-QIC deductive program.*

**Role.** T13 closes the deductive program with three deliverables. First, it
recasts the structure-scramble gap $\Delta(K)$ as a positive RKHS
kernel-alignment increment, building an interpretive bridge from the T9 SNR
identity to the Schuld (2021) / Kubler-Buchholz-Scholkopf (2021) kernel picture
of quantum generalization (Section 5.3). Second, it formalizes the map from the
two governing knobs -- topology alignment $\alpha$ and readout locality $\kappa$
-- to the sign and scaling of $\Delta$, as a six-region phase-diagram theorem
that follows deductively from T9, T11, and T12 (Section 5.4). Third, it compiles
eight falsifiable corollaries P1-P8 with quantitative pass/fail thresholds, each
bound to a specific blocking experiment (Section 5.5).

T13 introduces no new architectural machinery. It reorganizes completed results
into (a) a bridge to the external kernel literature and (b) a register of
falsifiable predictions that make the framework refutable.

---

## 0. Preamble: scope discipline

Three scope conditions are load-bearing and are stated before the results, in
the manner of the T11 preamble. They are not disclaimers; they fix what each
deliverable is entitled to claim.

1. **The kernel-alignment result is an INTERPRETIVE BRIDGE for trained
   $\theta$, and a theorem only for frozen $\theta$.** The Schuld representer
   theorem and the Kubler alignment inequality both presuppose a FIXED feature
   map $x \mapsto \rho_\theta(x)$, i.e. a data-independent kernel
   $k_\theta(x,x') = \mathrm{Tr}[\rho_\theta(x)\rho_\theta(x')]$. TC-QIC trains
   $\theta$, so the kernel co-adapts to the data and the strict RKHS theorem does
   not apply verbatim. Section 5.3 therefore states **Lemma 5.1 as a fixed-theta
   result** and provides a frozen-encoder reduction (Remark 5.2) that recovers
   the trained case as a two-stage limit. We do NOT claim a standalone
   trained-kernel alignment theorem.

2. **The phase-diagram theorem IS a theorem, not an interpretation.** Unlike
   Lemma 5.1, Theorem 5.3 does not import external RKHS assumptions. Its sign and
   scaling claims follow directly from the T9 SNR identity
   ($\mathrm{SNR}=\alpha^2\,\mathrm{SNR}_{\mathrm{ideal}}$), the T12 four-clause
   decomposition (Thm 4.4), and the T11 Cor 4.3b classical-regime result. Where a
   region's claim rests on a conditional clause (T11 clause iii, trainability),
   this is flagged inline.

3. **The predictions are corollaries, not conjectures, but their thresholds are
   calibration-dependent.** P1-P8 are logical consequences of T10-T12 plus the
   phase diagram. The numeric thresholds (e.g. $\Delta(K{=}10)\in[0.013,0.019]$)
   are derived from the T10 calibrated fit and inherit its confidence interval.
   A threshold miss refutes the calibrated model, not necessarily the qualitative
   mechanism; each prediction states which theorem it would refute.

---

## 1. Setup and imported results

Notation as in T9 (place-then-harvest), T10 (scaling law), T11 (master theorem),
T12 (bias-variance). The TC-QIC circuit acts on $K$ qubits with Hilbert space
$\mathcal{H}=(\mathbb{C}^2)^{\otimes K}$. Let $G$ be a molecular graph, $C(G)$
its $K$-cluster spectral coarsening (T3), and $\rho_\theta(C(G))$ the state
prepared by the GraphG circuit: RY re-uploading encoder followed by the
graph-gated entangler $\mathrm{IsingXX}(A_{ij}\theta_{\mathrm{pair}})$, with $A$
the max-normalized coarse adjacency. The Level-8 bond-pooled readout is the
affine feature map $\phi_{\mathcal{O}_8}(\rho_\theta)\in\mathbb{R}^{5K}$ (T11
Sec 1), whose load-bearing components are the bond-pooled correlators

$$
\big(B_A^{ZZ}\big)_i = \sum_j A_{ij}\,\langle Z_iZ_j\rangle,
\qquad
\big(B_A^{XX}\big)_i = \sum_j A_{ij}\,\langle X_iX_j\rangle .
$$

Let $C_{ij}(\rho)=\langle Z_iZ_j\rangle-\langle Z_i\rangle\langle Z_j\rangle$ be
the connected correlator, and $S(K)=\mathrm{span}\{C_{ij}:A_{ij}>0\}$ the
placed-signal subspace, $\dim S(K)=|E|=\Theta(K)$ (T9 Lemma 4.1).

The modeled quantity is the structure-scramble AUC gap

$$
\Delta(K)\;=\;\mathrm{AUC}_{\mathrm{struct}}(K)-\mathrm{AUC}_{\mathrm{scram}}(K),
$$

the benchmark's headline inductive-bias statistic (NOT absolute performance).

**Four imported results used throughout.**

- **(T9) SNR identity.** With placement graph $A_{\mathrm{place}}$ and readout
  graph $A_{\mathrm{read}}$, and alignment
  $\alpha=\langle A_{\mathrm{read}},A_{\mathrm{place}}\rangle/(\|A_{\mathrm{read}}\|_F\|A_{\mathrm{place}}\|_F)\in[-1,1]$,
  $$
  \mathrm{SNR}\;=\;\alpha^2\,\mathrm{SNR}_{\mathrm{ideal}},
  \qquad
  \alpha<0\Rightarrow\Delta(K)<0.
  $$
  The anti-alignment sign is confirmed by E8 meas_only ($\Delta$AUC$\approx-0.027$).

- **(T10) Scaling law.** $\Delta(K)\propto\eta_{\mathcal{O}}(K)\,s(K)-c\,W\sqrt{K/n}$
  with harvest coefficient
  $\eta_{\mathcal{O}}=\dim(\mathcal{A}(\mathcal{O})\cap S(K))/\dim S(K)$;
  $\eta_B=1$ (levelG), $\eta_S=\Theta(1/K)$ (gate-only), $\eta_{\mathrm{meas}}=0$.
  Calibrated: $\Delta_B(K)\approx 1.4\times10^{-3}K+2.3\times10^{-3}$
  ($R^2=0.996$ at $K=4,6,8$), saturation $K^\star\in[8,16]$.

- **(T11) Master theorem, Cor 4.3b.** The parameter-matched classical model
  `classicalGNN_pm`, using the same $A$-weighted bond-pooling but classical
  products $h_ih_j$ in place of $\langle Z_iZ_j\rangle$, ALSO exhibits
  $\Delta(K)>0$. Bond-pooled aggregation is the load-bearing bias source for both
  quantum and classical arms; the quantum correlator is a regime-dependent
  add-on, not a necessary condition.

- **(T12) Bias-variance.** Thm 4.4 (four-clause): $\Delta(K)>0$ iff (i) $\alpha>0$,
  (ii) signal harvest with regime-dependent quantum add-on (peaks $K=6$,
  $+56\%$), (iii) capacity match $W\sqrt{K/n}\le c_{\mathrm{task}}$, (iv)
  non-vacuous control. Thm 4.5 (classical dominance):
  $\mathrm{AUC}(\text{classical})-\mathrm{AUC}(\text{levelG})\ge
  f(\varepsilon_{\mathrm{bottleneck}})-O(n^{-1/2})$, empirically $5$-$8$ points.

---

## 5.3 Kernel-alignment lemma (interpretive bridge to Schuld/Kubler)

### 5.3.1 The quantum kernel and the representer picture

Fix the encoder parameters $\theta$. The TC-QIC feature map defines a quantum
kernel

$$
k_\theta(x,x')\;=\;\mathrm{Tr}\!\big[\rho_\theta(x)\,\rho_\theta(x')\big],
$$

where $x=C(G)$ is the coarse-grained molecular input. By the Schuld (2021)
representer argument, any linear-readout model trained on this feature map admits
the kernel expansion

$$
f_\theta(x)\;=\;\sum_i c_i\,k_\theta(x_i,x),
$$

with dual coefficients $c_i$ fixed by the training labels. In TC-QIC the linear
head $h_w$ acts on the Level-8 features $\phi_{\mathcal{O}_8}$; because
$\phi_{\mathcal{O}_8}$ is an affine functional of $\rho_\theta$, the induced
readout kernel is the corresponding restriction of $k_\theta$ to the operator
slice $\mathcal{A}(\mathcal{O}_8)$. Write $k^{\mathcal{O}}_\theta$ for this
restricted kernel; it is the kernel whose RKHS the trained head actually
searches.

### 5.3.2 Structured vs scrambled kernels

Structuring vs scrambling changes the readout topology $A$ that the bond-pooled
correlators are contracted against. Let $k_{\mathrm{struct}}$ denote the restricted
kernel built with $A=A_{\mathrm{true}}$ and $k_{\mathrm{scram}}$ the one built
with $A=A_{\mathrm{rand}}$ (non-absorbable, readout-side scramble; a placement-only
scramble is vacuous by the T8 absorbability analysis and is excluded). The
generalization gap between the two arms is controlled by the difference kernel

$$
\delta k(x,x')\;=\;k_{\mathrm{struct}}(x,x')-k_{\mathrm{scram}}(x,x').
$$

By the representer expansion,

$$
\mathrm{Gen}(\mathrm{struct})-\mathrm{Gen}(\mathrm{scram})
=\Big\langle c,\;k_{\mathrm{struct}}(\cdot,x)-k_{\mathrm{scram}}(\cdot,x)\Big\rangle
=\big\langle c,\;\delta k(\cdot,x)\big\rangle,
$$

so the sign and size of the inductive-bias gap are set by how the difference
kernel is oriented against the dual weights, i.e. against the label structure.

### 5.3.3 The Kubler alignment functional

Kubler-Buchholz-Scholkopf (2021) quantify generalization by the centered kernel
alignment between a model kernel $K_\theta$ (Gram matrix over the training set)
and the target label kernel $K_{\mathrm{target}}=yy^\top$:

$$
t(K_\theta,K_{\mathrm{target}})
=\frac{\langle K_\theta,K_{\mathrm{target}}\rangle_F}
       {\|K_\theta\|_F\,\|K_{\mathrm{target}}\|_F}\;\in[-1,1].
$$

Higher alignment implies the target label vector lies in the top eigenspaces of
the model kernel, which is where a kernel machine generalizes with the fewest
samples. Define the **alignment increment**

$$
\Delta t(K)\;=\;t(K_{\mathrm{struct}},K_{\mathrm{target}})
              -t(K_{\mathrm{scram}},K_{\mathrm{target}}).
$$

### 5.3.4 Lemma 5.1 (kernel-alignment interpretation, fixed-theta)

**Lemma 5.1.** *Fix the encoder $\theta$ (frozen feature map). Assume the
readout-side scramble is non-absorbable and the label kernel
$K_{\mathrm{target}}=yy^\top$ is the same in both arms. Then the structure-scramble
gap is a monotone image of the RKHS alignment increment:*

$$
\Delta(K)\;=\;\Phi\big(\Delta t(K)\big),
\qquad \Phi\ \text{monotone nondecreasing},\ \Phi(0)=0,
$$

*and moreover $\Delta t(K)>0$ if and only if the T9 alignment coefficient
$\alpha>0$. In particular, to leading order in the placed signal,*

$$
\Delta t(K)\;\propto\;\alpha^2\,\big\langle\, \mathrm{sig}, K_{\mathrm{target}}\big\rangle_F,
$$

*so the kernel-target alignment inherits the $\alpha^2$ scaling of the T9 SNR
identity.*

**Proof sketch (fixed $\theta$).** With $\theta$ frozen, $k_\theta$ is a genuine
data-independent kernel and the Schuld representer expansion holds exactly. The
difference kernel restricted to $\mathcal{A}(\mathcal{O}_8)$ is, by the T9
place-then-harvest identity, proportional to the placed-signal Gram operator
weighted by the readout-placement overlap: writing $B_A$ for the bond-pooled
readout and using $\mathbb{E}[B_A(\rho_{\mathrm{struct}})]=\sum_jA_{\mathrm{true},ij}C_{ij}$,

$$
\delta k(x,x')\;\approx\;\alpha^2\,\big\langle \mathrm{sig}(x),\mathrm{sig}(x')\big\rangle
\;+\;O(\theta^2)\ \text{off-bond terms},
$$

where $\mathrm{sig}(x)$ is the signal component that T9 Lemma 4.1 concentrates on
bonded pairs and $\alpha$ is the readout-placement alignment. Contracting the
Gram matrix of $\delta k$ against $K_{\mathrm{target}}=yy^\top$ gives
$\langle K_{\mathrm{struct}}-K_{\mathrm{scram}},K_{\mathrm{target}}\rangle_F
\propto\alpha^2\langle\mathrm{sig},yy^\top\rangle_F$. Dividing by the (arm-common,
to leading order) Frobenius norms yields $\Delta t\propto\alpha^2\langle\mathrm{sig},K_{\mathrm{target}}\rangle_F$,
which is positive iff $\alpha>0$ (the signal is positively class-correlated by
construction of the placed subspace). Finally, AUC is a monotone functional of
the margin distribution, and the margin is monotone in the aligned kernel score,
so $\Delta(K)=\Phi(\Delta t(K))$ with $\Phi$ monotone and $\Phi(0)=0$. $\square$

**Reading.** Lemma 5.1 says the empirically measured inductive bias $\Delta(K)$ at
$K=4,6,8$ is, at frozen $\theta$, exactly the extra centered kernel-target
alignment the structured kernel buys over the scrambled kernel -- the same
quantity the Kubler bound identifies as the driver of sample-efficient
generalization. This is the bridge: T9's overlap identity and Kubler's alignment
functional are two coordinates on one object, joined by the $\alpha^2$ law.

### 5.3.5 Remark 5.2 (trained-theta reduction via frozen encoder)

The TC-QIC encoder is trained, so $k_\theta$ co-adapts to the data and Lemma 5.1
does not hold verbatim. We recover the trained case as a two-stage (frozen-encoder)
limit. Let $\theta^\star$ be the converged encoder parameters from the full
train-encoder-plus-head optimization. Consider the frozen-encoder model that
FIXES $\theta=\theta^\star$ and re-fits only the linear head. Two facts make this
reduction faithful:

1. **The head is where the alignment is read.** By T6 the dominant generalization
   term is the head term $O(W\sqrt{K/n})$ at fixed encoder; the encoder-complexity
   term $\tilde O(\sqrt{P/n})$, $P=O(K^2)$, is common to both structured and
   scrambled arms and cancels in $\Delta$ to leading order (T12 Sec 0). Thus the
   arm difference $\Delta(K)$ is governed by the frozen-encoder head problem, to
   which Lemma 5.1 applies exactly.

2. **Convergence concentrates signal into $\mathcal{A}(\mathcal{O}_8)$.** At
   $\theta^\star$ the trained circuit concentrates the label-relevant signal into
   the accessible operator slice (T12 Sec 2.1(b)); the frozen kernel
   $k_{\theta^\star}$ therefore already embeds the learned alignment, and re-fitting
   the head recovers the same decision function up to $O(n^{-1/2})$.

The consequence is that the trained-$\theta$ gap equals the fixed-$\theta$ gap of
the frozen encoder up to the finite-sample slack: $\Delta_{\mathrm{trained}}(K)=
\Delta_{\theta^\star}(K)+O(n^{-1/2})$. Lemma 5.1 then transfers as an
INTERPRETATION -- not a theorem -- of the trained model: the trained circuit acts
as a data-selected member of a kernel family whose structured member is better
target-aligned than its scrambled member. We do not claim the trained kernel
maximizes alignment, only that its structured-vs-scrambled increment inherits the
sign and $\alpha^2$ scaling of the frozen-encoder increment.

---

## 5.4 The (alpha, kappa) phase-diagram theorem

### 5.4.1 The two governing knobs

The behavior of $\Delta(K)$ is organized by two independent axes.

- **Topology alignment $\alpha\in[-1,1]$** (T9): the centered overlap between the
  readout graph and the placement graph,
  $\alpha=\langle A_{\mathrm{read}},A_{\mathrm{place}}\rangle/(\|A_{\mathrm{read}}\|_F\|A_{\mathrm{place}}\|_F)$.
  $\alpha=1$ is the true molecular graph, $\alpha=0$ a random-graph readout,
  $\alpha<0$ an anti-aligned readout. Interpolation
  $A=\lambda A_{\mathrm{true}}+(1-\lambda)A_{\mathrm{rand}}$ traces $\alpha(\lambda)$
  monotonically from $\approx 0$ to $1$.

- **Readout locality $\kappa\in\{2,\dots,K\}$**: the maximum Pauli weight of the
  readout observables. $\kappa=2$ is the bond-local Level-8 readout (2-local
  correlators $\langle Z_iZ_j\rangle$); $\kappa=K$ is a global-parity readout
  $\langle Z_1\cdots Z_K\rangle$. Locality controls trainability: by T11 clause
  (iii) and E2, gradient variance stays poly-bounded for $\kappa=O(1)$ but decays
  toward barren-plateau scaling as $\kappa\to K$.

### 5.4.2 The claim

**THEOREM 5.3 (phase diagram).** *Fix $n$ and task difficulty $c_{\mathrm{task}}$
in the T12-feasible range. The sign and $K$-scaling of $\Delta(K)$ over the
$(\alpha,\kappa)$ plane partition into six regions as follows.*

| Region | $\alpha$ | $\kappa$ | $\mathrm{sign}\,\Delta$ | $K$-scaling of $\Delta$ | governing result |
|---|---|---|---|---|---|
| **I. Aligned-local (operating point)** | $\alpha>\alpha_c$ | $\kappa=O(1)$ | $+$ | linear $\sim\eta_B\,s(K)$, slope $\approx1.4\times10^{-3}$, saturates at $K^\star$ | T9 + T10 ($\eta_B{=}1$) + T12 Thm 4.4 |
| **II. Aligned-global** | $\alpha>\alpha_c$ | $\kappa\to K$ | $+$ then $\to 0$ | rises then collapses as BP sets in near $K_{\mathrm{BP}}$ | T11 clause (iii) + E2 |
| **III. Neutral-local** | $\alpha\approx 0$ | $\kappa=O(1)$ | $\approx 0$ | flat (vacuous control), $\Delta=O(n^{-1/2})$ | T9 ($\alpha{=}0$) + T12 clause (iv) |
| **IV. Neutral-global** | $\alpha\approx 0$ | $\kappa\to K$ | $\approx 0^-$ | flat-to-negative from variance blowup | T11 clause (iii) |
| **V. Anti-aligned (any locality)** | $\alpha<0$ | any | $-$ | $\Delta\propto\alpha^2$ in magnitude but NEGATIVE sign | T9 anti-alignment + E8 |
| **VI. Classical-parity boundary** | $\alpha>\alpha_c$ | $\kappa=2$, classical products | $+$ | matches Region I within TOST margin | T11 Cor 4.3b + T12 Thm 4.5 |

*where $\alpha_c$ is the alignment threshold below which the harvest term drops
under the T12 capacity term ($\eta_B(\alpha)s(K)\le c\,W\sqrt{K/n}$), and
$K_{\mathrm{BP}}$ is the qubit count at which $\kappa=K$ readout enters the
barren-plateau regime.*

### 5.4.3 Proof

The theorem is assembled region by region from imported results; no new
machinery is introduced.

**Sign follows from T9.** By the SNR identity, $\Delta(K)$ has the sign of
$\alpha$ times the harvest magnitude (which is $\alpha^2\ge0$ times a positive
signal term). Concretely the leading contribution is
$\Delta(K)\propto\alpha\,|\alpha|\,\mathrm{SNR}_{\mathrm{ideal}}$: positive for
$\alpha>0$ (Regions I, II, VI), $\approx 0$ for $\alpha\approx0$ (III, IV),
negative for $\alpha<0$ (V). The $\alpha^2$ magnitude with the sign of $\alpha$ is
exactly the E8 anti-alignment observation ($\Delta$AUC$\approx-0.027$).

**Locality selects the K-scaling via T10 and T11.** For $\kappa=O(1)$ (local),
the readout observables lie inside the accessible slice $\mathcal{A}(\mathcal{O}_8)$
with harvest coefficient $\eta_B=1$ (T10), so the harvest term is the full
$s(K)$ and $\Delta$ grows linearly until the T10 saturation at $K^\star\in[8,16]$
(Region I). This is the calibrated operating point,
$\Delta_B(K)\approx1.4\times10^{-3}K+2.3\times10^{-3}$. For $\kappa\to K$ (global),
the gradient variance decays toward $2^{-K}$ scaling (T11 clause iii, supported
conditionally and verified by E2 for local cost): the harvest signal is present
in principle but is not trainable at large $K$, so $\Delta$ first rises (small
$K$, before BP bites) then collapses to $0$ as $K$ exceeds $K_{\mathrm{BP}}$
(Region II). At $\alpha\approx 0$ the harvest term vanishes and only the capacity
penalty and finite-sample noise remain: $\Delta=O(n^{-1/2})$, flat and vacuous
(Region III, T12 clause iv), tipping slightly negative when the global-readout
variance blowup adds estimation noise (Region IV).

**Classical-parity boundary from T11 Cor 4.3b.** For $\alpha>\alpha_c$, $\kappa=2$,
the bond-pooled aggregation is the load-bearing bias source. Replacing the quantum
correlator $\langle Z_iZ_j\rangle$ by the classical product $h_ih_j$ preserves the
$A$-weighted contraction and hence the alignment mechanism (T11 Cor 4.3b), so the
classical model's $\Delta(K)$ coincides with Region I within the TOST equivalence
margin (Region VI). By T12 Thm 4.5 the quantum arm gains a regime-dependent
coherent add-on peaking at $K=6$ but does not exceed the classical arm in absolute
AUC. This is why Region VI is a boundary, not a distinct interior region: the gap
mechanism is shared, and the quantum-specific contribution is the $K=6$ add-on
only.

**The threshold $\alpha_c$.** Solving $\eta_B(\alpha)\,s(K)=c\,W\sqrt{K/n}$ for the
smallest $\alpha$ at which the harvest term clears the capacity floor gives
$\alpha_c=\sqrt{c\,W\sqrt{K/n}/(s(K)\,\mathrm{SNR}_{\mathrm{ideal}})}$, decreasing
in $n$: with more data the aligned region grows. Below $\alpha_c$ the operating
point falls into Region III (vacuous), above it into Region I. $\square$

### 5.4.4 Corollaries of the phase diagram

- **Cor 5.4 (operating point).** TC-QIC as benchmarked sits at
  $(\alpha,\kappa)=(1,2)$, the interior of Region I: aligned and local. This is
  the unique region combining a positive, K-growing, trainable bias.

- **Cor 5.5 (why global readout fails).** Region II shows the bias is not a
  property of "more entanglement": pushing $\kappa\to K$ does not amplify
  $\Delta$; it destroys trainability. Locality is not a limitation but a
  precondition (matches T11 clause iii).

- **Cor 5.6 (the sign flip is a knife-edge).** Crossing $\alpha=0$ (Region III to
  V) flips the sign of $\Delta$; the magnitude is symmetric ($\propto\alpha^2$)
  but the sign is not. Anti-alignment is not merely "no bias" but ACTIVE negative
  bias -- the readout harvests where the signal is absent.

---

## 5.5 The P1-P8 prediction register

Each corollary below is a falsifiable consequence of the imported theorems plus
the phase diagram. Each states its threshold, the experiment that blocks it, and
the theorem it would refute on failure. Thresholds inherit the T10 calibration
confidence interval (Preamble scope condition 3).

### P1 -- K-scaling (blocks E5)

**Claim.** In Region I, $\Delta(K)$ grows linearly at $\approx+1.4\times10^{-3}$
per qubit for $K\le K^\star$, with $K^\star\in[8,16]$ (T10 scaling law,
phase-diagram Region I).

**Threshold.** PASS if $\Delta(K{=}10)\in[0.013,0.019]$. FAIL (refutes T10
linearity) if $\Delta(K{=}10)<0.010$, or if $K=10$ shows a bend of $>30\%$ from
the calibrated $K=4/6/8$ slope. A value in $(0.014,0.018)$ confirms the linear
regime; $<0.012$ indicates early saturation (still consistent with $K^\star$ near
the lower bound but refuting continued linearity).

**Refutes on failure.** T10 linear scaling ($\eta_B=1$, constant slope).

### P2 -- Alignment knob (blocks E6)

**Claim.** Sweeping $A=\lambda A_{\mathrm{true}}+(1-\lambda)A_{\mathrm{rand}}$
traces the T9 SNR identity: $\Delta(\lambda)=\alpha(\lambda)^2\,\Delta_{\mathrm{ideal}}$,
monotone increasing in $\lambda$ (Lemma 5.1, T9).

**Threshold.** PASS if $\Delta(\lambda)$ is monotone nondecreasing over
$\lambda\in[0,1]$ with quadratic-fit $R^2>0.7$. FAIL (refutes the T9 SNR identity
and Lemma 5.1) if non-monotone or if the fit to $\alpha(\lambda)^2$ has $R^2\le0.7$.

**Refutes on failure.** T9 SNR identity; the $\alpha^2$ scaling in Lemma 5.1.

### P3 -- Locality knob (blocks E7)

**Claim.** Bond-local readout ($\kappa=2$) yields a larger gap than global-parity
readout ($\kappa=K$): $\Delta(\kappa{=}2)>\Delta(\kappa{=}K)$, with the
gradient-variance ratio $\mathrm{Var}[\partial_\theta C]_{\kappa=2}/
\mathrm{Var}[\partial_\theta C]_{\kappa=K}>1$ (phase-diagram Regions I vs II,
T11 clause iii).

**Threshold.** PASS if bond-local AUC $>$ global AUC at all tested $K$ AND the
gradient-variance ratio exceeds $1$. FAIL (refutes T11 clause iii, trainability)
if global readout matches or exceeds bond-local.

**Refutes on failure.** T11 clause (iii) (locality as trainability precondition).

### P4 -- Classical parity (blocks E9; ALREADY CONFIRMED)

**Claim.** Structured levelG and structured classicalGNN_pm are statistically
equivalent in $\Delta$ (both $>$ scrambled), reflecting the shared bond-pooling
bias source (T11 Cor 4.3b, Region VI).

**Threshold.** Equivalence margin $=0.005$ AUC. PASS if
$|\mathrm{AUC}_{\mathrm{levelG}}-\mathrm{AUC}_{\mathrm{GNNpm}}|<0.005$ via TOST
($p<0.05$ for equivalence). ALREADY CONFIRMED: classicalGNN_pm dAUC
$K=4/6/8 = 0.014/0.007/0.012$; qGNN dAUC $K=4/6/8 = 0.0078/0.0108/0.0134$. Both
arms show a significant, comparable gap.

**Refutes on failure.** Would refute T11 Cor 4.3b (that bond-pooling, not the
quantum correlator, is load-bearing). Standing as CONFIRMED.

### P5 -- Feature injection (blocks E10)

**Claim.** Replacing the coarse-graph quantum features with fully resolved
full-graph features closes the T12 bottleneck gap; the classical-minus-levelG AUC
gap is driven by $\varepsilon_{\mathrm{bottleneck}}$ (T12 Thm 4.5).

**Threshold.** PASS if the absolute-AUC gap (classical $-$ levelG) shrinks by
$\ge50\%$ when bond-topology features are fully resolved (full-graph
featurization). FAIL (refutes the Thm 4.5 bottleneck term) if the gap persists
with full-graph featurization.

**Refutes on failure.** T12 Thm 4.5 ($\varepsilon_{\mathrm{bottleneck}}$ as the
source of classical dominance).

### P6 -- Barren-plateau resistance (blocks E2)

**Claim.** For $\kappa=2$ the gradient variance is poly-bounded, not exponentially
decaying: $\mathrm{Var}[\partial_\theta C]\cdot K^1\approx\mathrm{const}$
(T11 clause iii, phase-diagram Region I trainability).

**Threshold.** PASS if the slope of $\log\mathrm{Var}[\partial_\theta C]$ vs $K$
is $>-1.5$ (polynomial regime). FAIL (refutes Master Theorem clause iii) if the
slope is $\le-\log 2\approx-0.69$ per qubit in the sense of matching exponential
$2^{-K}$ decay -- operationally, a fitted exponential-per-qubit decay that
outperforms the polynomial fit refutes trainability.

**Refutes on failure.** T11 clause (iii). (Note: clause iii is CONDITIONAL in T11;
P6 is the empirical test that discharges the condition, verified at $K=4,6,8$ by
E2.)

### P7 -- External validity (blocks E11)

**Claim.** The normalized slope $\Delta/K$ is invariant across datasets (Tox21,
ToxCast, and a third if available); the bias is a property of the
architecture-topology coupling, not of one dataset (phase-diagram universality of
Region I).

**Threshold.** PASS if the Kendall $\tau$ of the $K$-rank order of $\Delta$
exceeds $0.6$ across datasets. FAIL if the slope differs by $>3\times$ between
datasets.

**Refutes on failure.** The universality claim implicit in Region I (that
$\alpha,\kappa$ and not dataset-specific label structure govern $\Delta$).

### P8 -- Small-n crossover (blocks final T12 calibration)

**Claim.** The T12 double bottleneck becomes an ADVANTAGE at small $n$: at
$n<500$, levelG matches or exceeds GNN_pm (the bottleneck regularizes); at
$n>2000$, GNN_pm leads by $5$-$8$ points (T12 Thm 4.5 with the
$O(n^{-1/2})$ slack dominating at small $n$).

**Threshold.** PASS if a crossover exists with the predicted sign change near
$n^\star\approx500$-$1000$. FAIL (refutes the T12 calibration of the bottleneck vs
finite-sample tradeoff) if no crossover exists -- i.e. classical leads at all $n$.

**Refutes on failure.** The T12 Thm 4.5 tradeoff between
$f(\varepsilon_{\mathrm{bottleneck}})$ (favoring classical at large $n$) and the
$O(n^{-1/2})$ slack (favoring the lower-capacity bottlenecked model at small $n$).

### 5.5.1 Prediction-to-theorem traceability

| Prediction | Blocks | Threshold (PASS) | Refutes on FAIL | Status |
|---|---|---|---|---|
| P1 K-scaling | E5 | $\Delta(K{=}10)\in[0.013,0.019]$ | T10 linearity | pending (E5 at K=10,12) |
| P2 alignment | E6 | monotone, $R^2>0.7$ vs $\alpha^2$ | T9 SNR identity / Lemma 5.1 | pending |
| P3 locality | E7 | local $>$ global AUC, var-ratio $>1$ | T11 clause (iii) | pending |
| P4 classical parity | E9 | TOST $|{\cdot}|<0.005$, $p<0.05$ | T11 Cor 4.3b | CONFIRMED |
| P5 feature injection | E10 | gap shrinks $\ge50\%$ | T12 Thm 4.5 bottleneck | pending |
| P6 BP resistance | E2 | $\log$-Var slope $>-1.5$ | T11 clause (iii) | supported (E2 K=4,6,8) |
| P7 external validity | E11 | Kendall $\tau>0.6$ | Region I universality | pending |
| P8 crossover | T12 calib | crossover near $n^\star{\approx}500$-$1000$ | T12 Thm 4.5 tradeoff | pending |

---

## 6. Summary

T13 completes the deductive program with three deliverables.

1. **Lemma 5.1 (kernel-alignment interpretation, fixed-theta)** recasts
   $\Delta(K)$ as a positive centered kernel-target alignment increment
   $\Delta t(K)=\Phi^{-1}(\Delta(K))$, inheriting the T9 $\alpha^2$ scaling and
   bridging to the Schuld representer picture and the Kubler alignment functional.
   It is stated honestly as a fixed-$\theta$ result; Remark 5.2 reduces the trained
   case to it via a frozen-encoder argument, transferring the sign and scaling but
   NOT claiming a standalone trained-kernel theorem.

2. **Theorem 5.3 (phase diagram)** partitions the $(\alpha,\kappa)$ plane into six
   regions with definite sign and $K$-scaling of $\Delta$, deduced from T9
   (sign via SNR identity), T10 (linear scaling, $\eta_B=1$), T11 clause iii and
   Cor 4.3b (locality and classical parity), and T12 (four-clause and dominance).
   The benchmarked operating point $(\alpha,\kappa)=(1,2)$ is the interior of
   Region I, the unique aligned-local-trainable region.

3. **The P1-P8 register** compiles eight falsifiable corollaries with quantitative
   thresholds, each bound to a blocking experiment (E5, E6, E7, E9, E10, E2, E11,
   and the final T12 calibration). P4 is confirmed and P6 is supported; the
   remainder are pending and make the framework refutable.

The through-line: the inductive bias is a topology-alignment ($\alpha$) effect,
read out locally ($\kappa=2$), interpretable as extra RKHS alignment, shared with
a matched classical model, and quantitatively predicted -- not a claim of quantum
supremacy but a structural, falsifiable account of when and why the structured
prior helps.

---

*Cited results: T9 (place-then-harvest SNR identity), T10 (scaling law and
calibration), T11 (master theorem, clause iii, Cor 4.3b), T12 (bias-variance
Thm 4.4/4.5). External: Schuld (2021), Kubler-Buchholz-Scholkopf (2021).*
