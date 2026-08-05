# T5: The Q-IB Lagrangian -- Auto-Compression via Operator Geometry

*Status: COMPLETE | Priority: MED | Deps: T1, T2*

**Theorem (Auto-compression, informal).** Under the fixed Level-8 readout
$\mathcal{O}_8$, the compression term of the Q-IB Lagrangian is bounded by the
*operator geometry* alone, independently of the training parameters $\theta$ and
of the tradeoff parameter $\beta$:

$$
I\big(X;\, T_{\mathcal{O}_8}\big) \;=\; O(K \log K)
\qquad \text{(and } \le K \text{ bits exactly, via T1).}
$$

An unrestricted quantum readout would admit up to $\Theta(4^K)$ bits of
operator-space freedom; the Level-8 architecture caps the compression term at a
$\Theta(K)$-dimensional slice *before any optimization*. The Q-IB problem is
therefore automatically in a strong-compression regime: all of $\theta$ is spent
maximizing the relevance term $I(T_{\mathcal{O}_8}; Y)$ inside a pre-compressed
$O(K \log K)$-bit channel. This document assembles T1 (Holevo ceiling
$I(X; T_{\mathcal{O}}) \le \chi$) and T2 ($\dim \mathcal{A}(\mathcal{O}_8) = 5K$)
into that statement.

---

## 1. The Q-IB Lagrangian (Definition 1.4, assembled)

Recall the objects of T1. Let $x \mapsto \rho_\theta(x) \in \mathcal{D}(\mathcal{H})$
be the $K$-qubit quantum encoder ($\mathcal{H} = (\mathbb{C}^2)^{\otimes K}$), let
$X$ be the (discrete) input drawn from $p(x)$ on a finite alphabet $\mathcal{X}$,
let $Y$ be the target with Markov structure $Y \to X \to \rho_\theta(X) \to
T_{\mathcal{O}}$, and let $T_{\mathcal{O}}$ be the classical readout produced by
the fixed observable family $\mathcal{O}$ through the affine feature map

$$
\phi_{\mathcal{O}}(\rho) \;=\; \big(\mathrm{Tr}[O_1 \rho], \dots, \mathrm{Tr}[O_m \rho]\big)
\;\in\; \mathbb{R}^m,
\qquad O_a \in \mathcal{O}.
$$

**Definition 1.4 (Q-IB Lagrangian).**

$$
\mathcal{L}_{\text{Q-IB}}(\theta, \mathcal{O})
\;=\; \underbrace{I(X; T_{\mathcal{O}})}_{\text{compression}}
\;-\; \beta\, \underbrace{I(T_{\mathcal{O}}; Y)}_{\text{relevance}},
\qquad \beta > 0,
$$

optimized as

$$
\min_{\theta}\; \mathcal{L}_{\text{Q-IB}}(\theta, \mathcal{O}),
\qquad \mathcal{O} = \mathcal{O}_8 \text{ fixed by the architecture.}
$$

**The decisive structural difference from the classical IB.** The classical
Tishby-Pereira-Bialek bottleneck (Definition 1.0 of T1) minimizes over the
*entire* simplex of stochastic maps $p(t \mid x)$:

$$
\min_{p(t \mid x)}\; I(T; X) - \beta\, I(T; Y).
$$

The channel is *free*: the optimizer may allocate the compression budget
$I(T;X)$ arbitrarily among all measurable functions of $x$. In the Q-IB the
trainable object is the encoder $\theta$ *only*. The map from the state to the
representation, i.e. the measurement structure $\mathcal{O}$, is **not** a
variable of the learning problem -- it is architecture.

$$
\boxed{\;\textbf{The choice of } \mathcal{O} \textbf{ IS the inductive bias.}\;}
$$

It is the concrete restriction of the classical IB feasibility set, entering as
a *hard constraint* rather than a regularizer. T5 quantifies how severe that
constraint is: it bounds the compression term structurally.

---

## 2. The auto-compression theorem

Throughout, "bits" means logarithms base 2 (as in T1, Definition 1.3).

### 2.1 The geometric box confining the readout

By T2, the Level-8 family is
$\mathcal{O}_8 = \{X_i, Y_i, Z_i\}_{i=1}^K \cup \{O^{ZZ}_i\}_{i=1}^K \cup
\{O^{XX}_i\}_{i=1}^K$, with $m = 5K$ observables, and the feature vector is

$$
\phi_{\mathcal{O}_8}(\rho) \;=\;
\big(\; \underbrace{\langle X_i\rangle, \langle Y_i\rangle, \langle Z_i\rangle}_{3K \text{ single-qubit}},\;
\underbrace{\textstyle\sum_j A_{ij}\langle Z_iZ_j\rangle}_{K \text{ pooled } ZZ},\;
\underbrace{\textstyle\sum_j A_{ij}\langle X_iX_j\rangle}_{K \text{ pooled } XX}\;\big)
\;\in\; \mathbb{R}^{5K},
$$

where $\langle O \rangle = \mathrm{Tr}[O \rho]$.

**Lemma 2.1 (Bounded features; imported from `tc_qic_theory.md` sec. 2.2).**
Every single-qubit Pauli expectation lies in $[-1, 1]$. With max-normalized
bond weights ($\max_{ij} A_{ij} \le 1$) and per-cluster degree bounded by
$\deg_{\max} := \max_i \sum_{j:(i,j)\in E} |A_{ij}| \le \bar d$, each bond-pooled
coordinate satisfies $|b[i]| = |\sum_j A_{ij} \langle O_i O_j\rangle| \le
\deg_{\max}$. Hence the feature vector is confined to the axis-aligned box

$$
\phi_{\mathcal{O}_8}(\rho) \;\in\;
\mathcal{B} \;:=\; [-1, 1]^{3K} \times [-\deg_{\max}, \deg_{\max}]^{2K}
\;\subset\; \mathbb{R}^{5K},
\tag{2.1}
$$

with Euclidean radius $\|\phi_{\mathcal{O}_8}(\rho)\|_2 \le
\sqrt{3K + 2K\,\deg_{\max}^2} = \Theta(\sqrt{K})$ (for bounded degree
$\bar d = O(1)$). The box $\mathcal{B}$ has finite Lebesgue volume

$$
\mathrm{vol}(\mathcal{B}) \;=\; 2^{3K}\,(2\,\deg_{\max})^{2K}.
\tag{2.2}
$$

*Note that $\mathcal{B}$ depends only on $\mathcal{O}_8$ and the graph weights
$A$, not on $\theta$: whatever the encoder does, the readout cannot escape this
$5K$-dimensional box.*

### 2.2 Compression bound via bounded-support entropy

We give the geometric argument at a fixed finite measurement resolution
$\varepsilon > 0$ (physically: finite-shot estimation resolves each coordinate
only to $O(1/\sqrt{N_{\text{shots}}})$; mathematically, quantize each coordinate
onto an $\varepsilon$-grid). Let $Q_\varepsilon$ denote coordinatewise
quantization at resolution $\varepsilon$, and write $\tilde T := Q_\varepsilon(T_{\mathcal{O}_8})$.

**Proposition 2.2 (Geometric compression bound).** For any encoder $\theta$ and
any resolution $\varepsilon > 0$,

$$
I\big(X;\, \tilde T\big)
\;\le\; H(\tilde T)
\;\le\; \log_2 \frac{\mathrm{vol}(\mathcal{B})}{\varepsilon^{5K}}
\;=\; 3K \log_2\!\frac{2}{\varepsilon} + 2K \log_2\!\frac{2\,\deg_{\max}}{\varepsilon}.
\tag{2.3}
$$

*Proof.* $\tilde T$ is a discrete random variable supported on the grid points
of $Q_\varepsilon$ that fall inside $\mathcal{B}$ (by (2.1) the readout never
leaves $\mathcal{B}$). The number of such cells is at most
$\mathrm{vol}(\mathcal{B})/\varepsilon^{5K}$: each occupied cell has volume
$\varepsilon^{5K}$ and the cells are disjoint and contained in $\mathcal{B}$
(up to a boundary layer, absorbed into the inequality). For any discrete
variable on $N$ outcomes, $H(\tilde T) \le \log_2 N$, giving the second
inequality. The first is the standard bound $I(X;\tilde T) = H(\tilde T) -
H(\tilde T \mid X) \le H(\tilde T)$, valid because $H(\tilde T \mid X) \ge 0$
for *discrete* $\tilde T$ (the quantization is what makes the conditional
entropy nonnegative -- see the caveat below). $\blacksquare$

**Scaling.** At any *fixed* readout resolution $\varepsilon$, the pooled term
contributes $2K \log_2(2\deg_{\max}/\varepsilon)$. In the worst case
$\deg_{\max} \le K - 1$ (a hub cluster bonded to all others), so
$\log_2 \deg_{\max} = O(\log K)$ and

$$
\boxed{\; I\big(X;\, T_{\mathcal{O}_8}\big) \;=\; O(K \log K) \;}
\tag{2.4}
$$

as claimed. Under the physically relevant bounded-degree regime
$\deg_{\max} = \bar d = O(1)$ (molecular clusters have $O(1)$ chemical bonds),
the bound sharpens to $O(K)$ -- linear in the qubit count. Either way the
compression term is $\Theta(K \cdot \mathrm{polylog}\,K)$, a *polynomial* budget.

**The exponential contrast.** An unrestricted quantum readout reads the full
$\dim \mathrm{Herm}(\mathcal{H}) = 4^K$-dimensional operator space (T2, sec. 1).
Its feature vector lives in a $4^K$-dimensional box; the identical argument
gives a compression ceiling of $\Theta(4^K \log(1/\varepsilon))$ -- exponential
in $K$. The Level-8 projection replaces the exponent by $5K$: the compression
term is capped at a $\Theta(K)$-dimensional slice of an exponentially large
space, purely by the operator geometry of $\mathcal{O}_8$.

### 2.3 The airtight Holevo cap (from T1)

The geometric bound (2.3) is resolution-dependent by construction; the
information-theoretic ceiling from T1 is resolution-free and holds for the
continuous readout exactly. By Lemma 1.1 and Section 5 of T1 (affine-readout
DPI chain $X \to c \to \phi_{\mathcal{O}}$),

$$
I\big(X;\, T_{\mathcal{O}_8}\big)
\;\le\; \chi\big(\{p(x), \rho_\theta(x)\}\big)
\;\le\; \log_2 \dim \mathrm{supp}(\bar\rho)
\;\le\; K \text{ bits.}
\tag{2.5}
$$

The two bounds are complementary. (2.5) is the *quantum* ceiling: no measurement
whatsoever extracts more than $\chi \le K$ bits from the state ensemble. (2.4)
is the *operator-geometry* refinement at the readout level: even the accessible
$\chi$ bits are funneled through a $5K$-dimensional feature box whose
information content scales as $O(K \log K)$ at fixed resolution. Both are
$O(K\,\mathrm{polylog}\,K)$; neither depends on $\theta$. Together they pin the
compression term to a polynomial budget:

$$
I\big(X;\, T_{\mathcal{O}_8}\big)
\;\le\; \min\Big\{\,K,\;\; O(K \log K) \text{ at resolution } \varepsilon \,\Big\}
\;=\; O(K).
$$

**Caveat (the continuous idealization).** In the infinite-shot,
noise-free limit $\phi_{\mathcal{O}_8}$ is a *deterministic* function of the
discrete $X$, so $T_{\mathcal{O}_8}$ is discrete with $\le |\mathcal{X}|$ values
and $I(X; T_{\mathcal{O}_8}) = H(T_{\mathcal{O}_8}) \le \log_2 |\mathcal{X}|$;
the differential-entropy inequality $I = h(T) - h(T\mid X) \le h(T)$ is *not*
valid there, because $h(T \mid X) = -\infty$ for a deterministic map. This is
why Proposition 2.2 is stated for the quantized (equivalently finite-shot)
readout, where $H(\tilde T \mid X) \ge 0$ is automatic, and why the airtight
$\theta$-independent statement of record is the Holevo cap (2.5). Both routes
deliver the same qualitative conclusion: the compression term is polynomial in
$K$, not exponential.

---

## 3. Strong-compression regime: interpretation

In the classical IB, *strong compression* -- a small value of $I(T;X)$ -- is
something the optimizer must *earn*: one drives $\beta$ down, or explicitly
regularizes, and searches the channel simplex for a low-information
representation. Nothing structural prevents the classical channel from storing
all $H(X)$ bits about the input.

In TC-QIC strong compression is **free**. The $O(K \log K)$ ceiling of Section 2
is imposed by the $\Theta(K)$-dimensional measurement architecture and holds

- for *every* $\theta$ (the box $\mathcal{B}$ and the slice
  $\mathcal{A}(\mathcal{O}_8)$ are $\theta$-independent), and
- for *every* $\beta$ (the ceiling is a constraint on the feasible set, not a
  term in the objective).

This is the formal content of the slogan **"the double bottleneck prevents
over-smoothing."** The two bottlenecks are (i) the Holevo/measurement ceiling
$\chi \le K$ of T1 and (ii) the operator-geometry projection
$\dim \mathcal{A}(\mathcal{O}_8) = 5K$ of T2. Stacked, they guarantee the
circuit *cannot* store more than $O(K \log K)$ bits about $X$ in
$T_{\mathcal{O}_8}$. Overfitting to task-irrelevant molecular variation -- the
mechanism by which an over-expressive readout memorizes the training inputs --
is *structurally impossible*: there is no channel capacity in which to store the
irrelevant bits. Compression is not tuned; it is a property of the architecture.

By Prop. 1.2 of T2, the invisible component $\Pi_{\mathcal{A}^\perp}\rho$ (of
dimension $4^K - 5K$) is exactly annihilated by the readout, so any input
variation living in $\mathcal{A}^\perp$ contributes *zero* to $I(X; T_{\mathcal{O}_8})$.
The compression is a hard projection, and (2.4) is its information-theoretic
shadow.

---

## 4. The $\beta$ parameter and the Pareto tradeoff

With the compression term pinned by architecture, the optimization
$\min_\theta \mathcal{L}_{\text{Q-IB}}(\theta, \mathcal{O}_8)$ reduces to a
constrained relevance-maximization:

$$
\max_{\theta}\; I\big(T_{\mathcal{O}_8}; Y\big)
\quad \text{subject to} \quad
I\big(X; T_{\mathcal{O}_8}\big) = O(K \log K)
\text{ (automatic).}
$$

The learner spends its freedom $\theta$ **rotating the encoded state so that
task-relevant structure lands inside $\mathcal{A}(\mathcal{O}_8)$** -- i.e.
maximizing the second term within the pre-compressed slice. Varying $\beta$
traces a Pareto front $\big(I(X;T_{\mathcal{O}_8}),\, I(T_{\mathcal{O}_8};Y)\big)$,
but that front lies *entirely inside* the $O(K \log K)$-compressed regime:

- the circuit cannot move to arbitrarily *high* compression by luck -- it is
  already compressed to $O(K \log K)$, and $\beta$ only shifts *where within the
  slice* the relevance mass concentrates;
- the circuit cannot escape to an *overcomplete* representation -- the range of
  the readout is fixed at $\dim \mathcal{A}(\mathcal{O}_8) = 5K$ by T2, so no
  setting of $\theta$ or $\beta$ enlarges the accessible operator subspace.

Thus $\beta$ is a *soft dial within a hard box*: it selects a point on the
Pareto curve, but the curve itself is confined to the compressed regime by the
architecture. This is qualitatively unlike the classical IB, where the Pareto
curve spans the full range from $I(T;X) = 0$ to $I(T;X) = H(X)$.

---

## 5. Connection to Canatar et al. 2022

Canatar, Peters, Pehlevan, Wild, and Kubler (2022) control quantum-kernel
generalization through a *bandwidth* hyperparameter $c$ that rescales the
encoding rotation angles and thereby tunes the decay of the kernel's eigenvalue
spectrum. Small bandwidth concentrates the spectral mass on low-frequency
(task-relevant) modes and damps high-frequency (task-irrelevant) modes -- a
*soft, parametric* auto-compression: the misaligned modes retain small but
nonzero eigenvalue, and the amount of compression is a continuous function of
$c$ that the practitioner must tune.

TC-QIC reaches strong compression by a different route. The measurement channel
$\phi_{\mathcal{O}_8}$ is a *hard projection* onto $\mathcal{A}(\mathcal{O}_8)$
(T2, Prop. 1.2): the $4^K - 5K$ misaligned operator modes are assigned
eigenvalue *exactly zero*, not merely damped. The compression is therefore

| | Canatar et al. 2022 | TC-QIC Level-8 (T5) |
|---|---|---|
| Compression mechanism | soft spectral decay | hard operator projection |
| Control | bandwidth $c$ (parametric knob) | readout $\mathcal{O}_8$ (structural) |
| Irrelevant modes | small nonzero eigenvalue | exactly zero eigenvalue |
| Compression budget | tuned via $c$ | fixed at $O(K \log K)$ by geometry |
| Origin of the prior | kernel hyperparameter | molecular bond graph $\mathrm{Pool}_A$ |

Both are auto-compression mechanisms and both cap the effective information
about $X$. The distinction is that Canatar et al.'s cap is *parametric* (a
continuous function of $c$ the learner must set) while TC-QIC's is *structural*
(a fixed consequence of the $\Theta(K)$-dimensional measurement, independent of
any hyperparameter and of $\beta$). T5 is the operator-space, measurement-level
statement of what their bandwidth achieves at the kernel-spectrum level.

---

## 6. Limitation: the bound is loose

The bound $I(X; T_{\mathcal{O}_8}) = O(K \log K)$ (and its $O(K)$ Holevo
sharpening) is a *worst-case ceiling*: it counts the maximal information a
$5K$-dimensional bounded feature vector *could* carry. For a *trained* model the
actual mutual information depends on how much the encoder varies $\rho_\theta(x)$
across molecules -- i.e. how much of the $5K$-dimensional box is actually
populated.

The empirical operator-geometry study E1 (`E1_operator_geometry_results.md`)
measures the *effective rank* of the realized feature covariance and finds
$\mathrm{eff\_rank}/K \approx 1.15\text{--}1.86$ (mean $\approx 1.5$) across
$K \in \{4, 6, 8\}$: the trained circuit populates only $\approx 1.5 K$ of the
$5K$ available feature dimensions with significant variance. The *practical*
compression is therefore even tighter than the $O(K \log K)$ theoretical
ceiling -- the realized $I(X; T_{\mathcal{O}_8})$ sits well below the box-volume
bound because the encoder concentrates its output on a low-effective-rank
sub-box.

Two directions sharpen T5:

1. **T13 (empirical MI).** Estimate $I(X; T_{\mathcal{O}_8})$ directly with a
   KSG (Kraskov-Stoegbauer-Grassberger) nearest-neighbor estimator on the
   realized $(\phi_{\mathcal{O}_8}(x), x)$ pairs, and compare against the
   $O(K \log K)$ ceiling and the E1 effective-rank prediction.
2. **Tightening via effective rank.** Replace $5K$ by the realized effective
   dimension $r_{\text{eff}} \approx 1.5K$ in the box-volume argument (2.3),
   yielding a data-dependent bound $O(r_{\text{eff}} \log K)$ that tracks the
   trained model rather than the worst case.

Neither refinement changes the qualitative verdict of T5: the Q-IB compression
term is polynomial ($O(K \cdot \mathrm{polylog}\,K)$) in the qubit count, not
exponential -- and it is so *automatically*, fixed by the operator geometry of
$\mathcal{O}_8$ before training and independently of $\beta$.

---

## References

- N. Tishby, F. Pereira, W. Bialek (1999). The information bottleneck method.
  *Proc. 37th Allerton Conf.*
- A. S. Holevo (1973). Bounds for the quantity of information transmitted by a
  quantum communication channel. *Probl. Peredachi Inf.* 9(3), 3-11.
- A. Canatar, E. Peters, C. Pehlevan, S. M. Wild, J. M. Kubler (2022).
  Bandwidth enables generalization in quantum kernel models.
  *arXiv:2206.06686* / TMLR 2023.
- A. Kraskov, H. Stoegbauer, P. Grassberger (2004). Estimating mutual
  information. *Phys. Rev. E* 69, 066138.
- T. M. Cover, J. A. Thomas (2006). *Elements of Information Theory*, 2nd ed.,
  Ch. 8-9 (differential entropy, maximum-entropy bounds).
- Companion notes: `T1_holevo_qib.md` (Holevo ceiling, measurement
  bottleneck), `T2_operator_geometry.md` (accessible subspace dimension
  $\dim \mathcal{A}(\mathcal{O}_8) = 5K$), `tc_qic_theory.md` sec. 2.2
  (Lemma 2.1, bounded features), `E1_operator_geometry_results.md` (effective
  rank).
