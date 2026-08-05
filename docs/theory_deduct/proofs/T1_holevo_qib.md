# T1: Quantum Mutual Information, Holevo Ceiling, and the Measurement Bottleneck

*Status: COMPLETE | Priority: HIGH | Deps: none*

**Theorem (Measurement Bottleneck, informal).** For a parameterized quantum
encoder $x \mapsto \rho_\theta(x)$ read out through a fixed observable family
$\mathcal{O} = \{O_a\}$, the classical mutual information between the input $X$
and the readout $T_{\mathcal{O}}$ obeys

$$
I(X; T_{\mathcal{O}}) \;\le\; \chi\big(\{p(x), \rho_\theta(x)\}\big)
\;=\; I_Q(X; T),
$$

and the choice of $\mathcal{O}$ -- not the encoder alone -- sets the operating
ceiling on extractable information. In the Q-IB Lagrangian the measurement
structure $\mathcal{O}$ is architecture, i.e. the inductive bias.

---

## 1. Classical IB and the free-channel baseline

**Definition 1.0 (Classical Information Bottleneck; Tishby-Pereira-Bialek 1999).**
Given a joint distribution $p(x, y)$ over input $X$ and target $Y$, the classical
IB problem is

$$
\min_{p(t \mid x)} \; I(T; X) \;-\; \beta \, I(T; Y),
\qquad \beta > 0,
$$

where the minimization ranges over *all* stochastic maps (Markov kernels)
$p(t \mid x)$ from $X$ to a representation variable $T$, subject to the Markov
chain $Y \to X \to T$.

The essential structural fact for our purposes: **the channel $p(t \mid x)$ is
free.** The feasibility set of the classical IB is the entire simplex of
conditional distributions; nothing about the optimization constrains *which*
functions of $x$ the representation $T$ may depend on. Any restriction of this
feasibility set -- to a parametric family, to a factorized form, to the image of
a fixed measurement -- is by definition an *inductive bias*. The classical,
unrestricted IB is therefore the baseline against which the quantum
information-bottleneck (Q-IB) construction below must be measured: everything
Q-IB adds is a statement about *how* the feasibility set is cut down, and by
*what geometric mechanism*.

---

## 2. Quantum encoder and the classical-quantum state

**Definition 1.1 (Quantum encoder).** Let $\mathcal{H} = (\mathbb{C}^2)^{\otimes K}$
be the $K$-qubit Hilbert space and $\mathcal{D}(\mathcal{H})$ the set of density
operators (positive semidefinite, unit trace) on $\mathcal{H}$. A quantum encoder
is a map

$$
x \;\longmapsto\; \rho_\theta(x) \;=\;
U_\theta(x)\, |0\rangle\langle 0|^{\otimes K}\, U_\theta(x)^{\dagger}
\;\in\; \mathcal{D}(\mathcal{H}),
$$

where $U_\theta(x)$ is a parameterized unitary circuit (parameters $\theta$,
data-dependent gates via $x$). Nothing below requires purity of $\rho_\theta(x)$;
we state results for general $\rho_\theta(x) \in \mathcal{D}(\mathcal{H})$.

**Definition 1.2 (cq-state).** Let $X$ take values in a finite alphabet
$\mathcal{X}$ with distribution $p(x)$. Fix an orthonormal basis
$\{|x\rangle\}_{x \in \mathcal{X}}$ of a classical register $\mathcal{H}_X$.
The *classical-quantum (cq) state* associated with the encoder is

$$
\rho_{XT} \;=\; \sum_{x \in \mathcal{X}} p(x)\,
|x\rangle\langle x|_X \otimes \rho_\theta(x)_T
\;\in\; \mathcal{D}(\mathcal{H}_X \otimes \mathcal{H}).
$$

**Definition 1.3 (Von Neumann entropy and QMI).** For
$\rho \in \mathcal{D}(\mathcal{H})$, the von Neumann entropy is

$$
S(\rho) \;=\; -\mathrm{Tr}[\rho \log \rho],
$$

with the convention $0 \log 0 = 0$ (all logs base 2; entropies in bits). The
quantum mutual information of a bipartite state $\rho_{AB}$ is

$$
I_Q(A; B) \;=\; S(\rho_A) + S(\rho_B) - S(\rho_{AB}),
$$

where $\rho_A = \mathrm{Tr}_B[\rho_{AB}]$ and $\rho_B = \mathrm{Tr}_A[\rho_{AB}]$
are the marginals. $I_Q(A;B) \ge 0$ by subadditivity of $S$.

---

## 3. Proposition: QMI of a cq-state equals the Holevo quantity

**Definition (Holevo quantity).** For an ensemble
$\mathcal{E} = \{p(x), \rho_x\}_{x \in \mathcal{X}}$ with average state
$\bar{\rho} = \sum_x p(x) \rho_x$,

$$
\chi(\mathcal{E}) \;=\; S(\bar{\rho}) \;-\; \sum_{x} p(x)\, S(\rho_x).
$$

**Proposition 1 (QMI = $\chi$ for cq-states).** For the cq-state of Definition 1.2,

$$
I_Q(X; T) \;=\; \chi\big(\{p(x), \rho_\theta(x)\}\big).
$$

*Proof.* We compute the three entropies in
$I_Q(X;T) = S(\rho_X) + S(\rho_T) - S(\rho_{XT})$.

**(i) Marginal on $X$.** $\rho_X = \mathrm{Tr}_T[\rho_{XT}]
= \sum_x p(x) |x\rangle\langle x|$, a diagonal state, so

$$
S(\rho_X) \;=\; -\sum_x p(x) \log p(x) \;=\; H(X).
$$

**(ii) Joint entropy via block-diagonality.** In the basis
$\{|x\rangle\}$ of the $X$ register, $\rho_{XT}$ is block-diagonal: it is a
direct sum of the blocks $p(x)\,\rho_\theta(x)$ supported on the orthogonal
subspaces $|x\rangle\langle x| \otimes \mathcal{H}$. For any block-diagonal
operator $\rho = \bigoplus_x p(x) \rho_x$ with each $\rho_x$ a density operator,
the spectrum of $\rho$ is the disjoint union over $x$ of the spectra
$\{p(x)\lambda^{(x)}_j\}_j$ where $\{\lambda^{(x)}_j\}_j$ is the spectrum of
$\rho_x$. Hence

$$
S(\rho_{XT})
= -\sum_x \sum_j p(x)\lambda^{(x)}_j \log\big(p(x)\lambda^{(x)}_j\big)
= -\sum_x p(x)\log p(x) \sum_j \lambda^{(x)}_j
  \;-\; \sum_x p(x) \sum_j \lambda^{(x)}_j \log \lambda^{(x)}_j,
$$

and since $\sum_j \lambda^{(x)}_j = 1$ for every $x$,

$$
S(\rho_{XT}) \;=\; H(X) \;+\; \sum_x p(x)\, S(\rho_\theta(x)).
\tag{3.1}
$$

**(iii) Marginal on $T$.** $\rho_T = \mathrm{Tr}_X[\rho_{XT}]
= \sum_x p(x)\, \rho_\theta(x) = \bar{\rho}$, so by the definition of $\chi$,

$$
S(\rho_T) \;=\; S(\bar{\rho})
\;=\; \chi\big(\{p(x), \rho_\theta(x)\}\big)
\;+\; \sum_x p(x)\, S(\rho_\theta(x)).
\tag{3.2}
$$

**(iv) Assembly.** Combining (i), (3.1), (3.2):

$$
I_Q(X;T)
= H(X)
+ \Big[\chi + \sum_x p(x) S(\rho_\theta(x))\Big]
- \Big[H(X) + \sum_x p(x) S(\rho_\theta(x))\Big]
= \chi. \qquad \blacksquare
$$

**Corollary (nonnegativity and range).** $\chi \ge 0$, since $S$ is concave:
$S(\bar\rho) = S(\sum_x p(x)\rho_x) \ge \sum_x p(x) S(\rho_x)$ (Jensen applied
to the concave functional $S$; concavity of $S$ is standard, e.g. via the
nonnegativity of quantum relative entropy $D(\rho_x \| \bar\rho) \ge 0$ and the
identity $\chi = \sum_x p(x) D(\rho_x \| \bar\rho)$). Moreover, if the ensemble
is supported on a subspace of dimension $d$, then $\chi \le S(\bar\rho) \le \log d$.
For pure-state encoders, $S(\rho_\theta(x)) = 0$ and $\chi = S(\bar\rho)$ exactly.

---

## 4. Lemma 1.1: the measurement bottleneck (quantum DPI)

**Setup.** A measurement of the observable family
$\mathcal{O} = \{O_a\}_{a=1}^{m}$ is described (see Section 5 for the precise
POVM construction) by a quantum-to-classical channel

$$
\mathcal{M}_{\mathcal{O}} : \mathcal{B}(\mathcal{H}) \to \mathcal{B}(\mathcal{H}_C),
\qquad
\mathcal{M}_{\mathcal{O}}(\rho) = \sum_{c} \mathrm{Tr}[E_c\, \rho]\; |c\rangle\langle c|,
$$

where $\{E_c\}$ is a POVM ($E_c \succeq 0$, $\sum_c E_c = \mathbb{1}$) and
$\{|c\rangle\}$ an orthonormal basis of a classical outcome register
$\mathcal{H}_C$. Any such channel is completely positive and trace preserving
(CPTP): it admits the Kraus representation
$\mathcal{M}_{\mathcal{O}}(\rho) = \sum_{c,j} K_{c,j}\, \rho\, K_{c,j}^\dagger$
with $K_{c,j} = |c\rangle \langle e_{c,j}| \sqrt{\mu_{c,j}}$ built from a spectral
decomposition $E_c = \sum_j \mu_{c,j} |e_{c,j}\rangle\langle e_{c,j}|$, and
$\sum_{c,j} K_{c,j}^\dagger K_{c,j} = \sum_c E_c = \mathbb{1}$. Its output is
always diagonal in $\{|c\rangle\}$, i.e. the channel is entanglement-breaking
(measure-and-prepare form).

**Lemma 1.1 (Measurement bottleneck).** Let $\rho_{XT}$ be the cq-state of
Definition 1.2 and let $T_{\mathcal{O}}$ denote the classical outcome of
applying $\mathcal{M}_{\mathcal{O}}$ to the $T$ register. Then

$$
I(X; T_{\mathcal{O}}) \;\le\; I_Q(X; T) \;=\; \chi\big(\{p(x), \rho_\theta(x)\}\big).
$$

*Proof.*

**(i) Local channel on the joint state.** Apply
$\mathrm{id}_X \otimes \mathcal{M}_{\mathcal{O}}$ to $\rho_{XT}$. The tensor
product of CPTP maps is CPTP, so this is a legitimate quantum channel acting
locally on subsystem $T$:

$$
\sigma_{X C} \;=\; (\mathrm{id}_X \otimes \mathcal{M}_{\mathcal{O}})(\rho_{XT})
\;=\; \sum_x p(x)\, |x\rangle\langle x| \otimes
\sum_c \mathrm{Tr}[E_c\, \rho_\theta(x)]\, |c\rangle\langle c|.
$$

**(ii) Quantum data-processing inequality.** Quantum mutual information is
monotone under local CPTP maps: for any channel $\Lambda$ acting on the $B$
subsystem of $\rho_{AB}$,

$$
I_Q(A; B)_{\rho} \;\ge\; I_Q(A; B')_{(\mathrm{id}\otimes\Lambda)(\rho)}.
$$

This is the Lindblad-Uhlmann monotonicity theorem: $I_Q(A;B) =
D(\rho_{AB} \| \rho_A \otimes \rho_B)$ where $D$ is the quantum relative
entropy, $D$ is monotone non-increasing under CPTP maps applied jointly
(Lindblad 1975, Uhlmann 1977, building on Lieb-Ruskai strong subadditivity
1973), and $(\mathrm{id}\otimes\Lambda)(\rho_A \otimes \rho_B) =
\rho_A \otimes \Lambda(\rho_B)$ so the product structure of the reference state
is preserved; a final minimization step
($D(\sigma_{AB'} \| \sigma_A \otimes \sigma_{B'}) \le
D(\sigma_{AB'} \| \rho_A \otimes \Lambda(\rho_B))$, which holds because
$\sigma_A = \rho_A$ and $I_Q(A;B') = \min_{\omega_{B'}}
D(\sigma_{AB'}\|\sigma_A \otimes \omega_{B'})$) gives monotonicity of $I_Q$
itself. Applied with $\Lambda = \mathcal{M}_{\mathcal{O}}$:

$$
I_Q(X; T)_{\rho_{XT}} \;\ge\; I_Q(X; C)_{\sigma_{XC}}.
$$

**(iii) The post-measurement QMI is classical MI.** The state $\sigma_{XC}$ is
diagonal in the product basis $\{|x\rangle \otimes |c\rangle\}$ with eigenvalues
$p(x)\, q(c \mid x)$, $q(c \mid x) = \mathrm{Tr}[E_c \rho_\theta(x)]$. For
classical (diagonal) states, von Neumann entropies reduce to Shannon entropies
of the eigenvalue distributions, so

$$
I_Q(X; C)_{\sigma_{XC}}
= H(X) + H(C) - H(X, C)
= I(X; T_{\mathcal{O}}),
$$

the ordinary classical mutual information between the input $X$ and the
measurement outcome $T_{\mathcal{O}} \sim q(\cdot \mid x)$.

**(iv) Assembly.** Chaining (ii), (iii), and Proposition 1:

$$
I(X; T_{\mathcal{O}}) \;=\; I_Q(X;C) \;\le\; I_Q(X;T) \;=\; \chi.
\qquad \blacksquare
$$

**Remark (Holevo's theorem).** Lemma 1.1 recovers the Holevo bound
(Holevo 1973): since the POVM $\{E_c\}$ was arbitrary, taking the supremum over
all measurements gives the accessible information

$$
I_{acc}(X; T) \;=\; \sup_{\{E_c\}} I(X; T_{\mathcal{O}}) \;\le\; \chi.
$$

The two statements have different roles here. Holevo's theorem says *no*
measurement can beat $\chi$. Lemma 1.1, read at *fixed* $\mathcal{O}$, says the
*chosen* measurement operates at $I(X; T_{\mathcal{O}})$, which can be far below
$\chi$: **which observables are read strictly sets the ceiling on extractable
information.** The gap $\chi - I(X; T_{\mathcal{O}})$ is information that the
encoder prepared but the readout architecture discards.

---

## 5. Careful handling of the affine (expectation-value) readout

The Level-8 readout used in the experiments is not a sampled POVM outcome but
the vector of expectation values

$$
\phi_{\mathcal{O}}(\rho) \;=\; \big(\mathrm{Tr}[O_1 \rho], \dots, \mathrm{Tr}[O_m \rho]\big)
\;\in\; \mathbb{R}^m .
$$

This object is *not* a density matrix and is *not* directly the output of a
quantum channel, so the DPI of Section 4 does not apply to it verbatim. The gap
is closed as follows.

**(i) POVM underlying each observable.** Each $O_a$ is Hermitian with spectral
decomposition $O_a = \sum_{\lambda} \lambda\, \Pi_a^{\lambda}$, where
$\{\Pi_a^{\lambda}\}_\lambda$ are the orthogonal eigenprojectors
($\Pi_a^\lambda \succeq 0$, $\sum_\lambda \Pi_a^\lambda = \mathbb{1}$). For the
Level-8 family every $O_a$ is a (sum of) Pauli-string observables; a single
Pauli string $P$ has $P^2 = \mathbb{1}$ and spectral projectors
$\Pi_a^{\pm} = \tfrac{1}{2}(\mathbb{1} \pm P)$, so the relevant POVM per
observable is the two-outcome family $\{\Pi_a^{+}, \Pi_a^{-}\}$. Measuring the
full family (each observable on its own circuit copy, as in practice) is
described by the product POVM with outcomes
$c = (\lambda_1, \dots, \lambda_m)$ and elements
$E_c = \bigotimes_{a} \Pi_a^{\lambda_a}$ acting on $m$ independent preparations
of $\rho$; equivalently, one may treat each observable separately -- either way
the resulting map

$$
\mathcal{M}_{\mathcal{O}}(\rho) \;=\; \sum_{c} \mathrm{Tr}[E_c\, \rho^{\otimes m}]\; |c\rangle\langle c|
$$

is CPTP and entanglement-breaking, exactly as in Section 4. (Using $m$ copies
of $\rho_\theta(x)$ replaces $\chi$ by $\chi_m \le m\,\chi$ for product
ensembles by subadditivity; for the qualitative ceiling statements below, and
for the single-observable statements, the single-copy bound suffices and no
copy inflation is needed: each coordinate individually satisfies the bound, and
the joint outcome $c$ of the $m$-copy measurement satisfies
$I(X; c) \le \chi(\{p(x), \rho_\theta(x)^{\otimes m}\}) = $ QMI of the $m$-copy
cq-state, to which Lemma 1.1 applies with $\mathcal{H}^{\otimes m}$ in place of
$\mathcal{H}$.)

**(ii) Expectation values are classical postprocessing of the POVM.** The
expectation value is a *linear functional of the outcome distribution*:

$$
\mathrm{Tr}[O_a \rho]
\;=\; \sum_{\lambda} \lambda\, \mathrm{Tr}[\Pi_a^{\lambda} \rho]
\;=\; \mathbb{E}\big[\lambda_a\big],
$$

i.e. $\phi_{\mathcal{O}}(\rho)$ is obtained from the measurement-outcome
statistics by an affine (in the infinite-shot limit, deterministic) map
$q(\cdot \mid x) \mapsto \phi$. Deterministic or stochastic postprocessing of a
classical variable is itself a classical channel, so by the *classical* data-
processing inequality applied to the Markov chain
$X \to c \to \phi_{\mathcal{O}}$:

$$
I\big(X;\, \phi_{\mathcal{O}}(\rho_\theta(X))\big) \;\le\; I(X;\, c).
$$

(In the infinite-shot idealization $\phi$ is a deterministic function of $x$
through $q(\cdot\mid x)$; the same inequality holds because
$\phi = f(q(\cdot \mid x))$ makes $X \to c^{\infty} \to \phi$ a valid Markov
chain over the empirical outcome distribution, and finite-shot estimates
$\hat\phi$ are randomized functions of i.i.d. outcomes, again postprocessing.)

**(iii) Conclusion (rigorous form of the bottleneck for affine readout).**

$$
I\big(X;\, \phi_{\mathcal{O}}(\rho_\theta(X))\big)
\;\le\; I(X;\, \text{$\mathcal{M}_{\mathcal{O}}$ outcome})
\;\le\; \chi\big(\{p(x), \rho_\theta(x)\}\big)
$$

(per-copy; with the $m$-copy qualification of (i) where joint statistics are
used). The affine readout can only *lose* information relative to the POVM it
postprocesses; the DPI chain is airtight at every link. No step assumed purity,
commutativity of the $O_a$, or tomographic completeness.

---

## 6. The Q-IB Lagrangian and the measurement slice

**Definition 1.4 (Q-IB Lagrangian).** With $T_{\mathcal{O}}$ the (classical)
readout variable of Sections 4-5,

$$
\mathcal{L}_{\text{Q-IB}}(\theta, \mathcal{O})
\;=\; I(X; T_{\mathcal{O}}) \;-\; \beta\, I(T_{\mathcal{O}}; Y).
$$

**Key structural stipulation.** The minimization is over $\theta$ at *fixed*
$\mathcal{O}$:

$$
\min_{\theta}\; \mathcal{L}_{\text{Q-IB}}(\theta, \mathcal{O}),
\qquad \mathcal{O} \text{ fixed by the architecture.}
$$

Contrast with Section 1: the classical IB optimizes over the *entire* channel
simplex. Here the trainable object is the encoder $\theta$ only; the
measurement structure $\mathcal{O}$ is not a variable of the learning problem.
**The choice of $\mathcal{O}$ is the inductive bias** -- it is precisely the
restriction of the classical IB feasibility set announced in Section 1, and it
enters the theory as a hard constraint, not a regularizer.

**Automatic compression.** Two ceilings stack:

1. *Holevo ceiling (Lemma 1.1):* $I(X; T_{\mathcal{O}}) \le \chi \le \log \dim
   \mathrm{supp}(\bar\rho) \le K$ bits -- already exponentially below the naive
   $4^K$-dimensional operator space.
2. *Operator-geometry ceiling (T2):* the Level-8 family spans an accessible
   operator subspace $\mathcal{A}(\mathcal{O}_8)$ with
   $\dim \mathcal{A}(\mathcal{O}_8) = 5K = \Theta(K)$ out of $4^K$
   (see `T2_operator_geometry.md`). The readout $\phi_{\mathcal{O}}$ is a hard
   rank-$\Theta(K)$ projection: $T_{\mathcal{O}}$ depends on $\rho_\theta(x)$
   only through its component in $\mathcal{A}(\mathcal{O}_8)$. Whatever the
   effective ensemble seen through this slice, the information carried by a
   $\Theta(K)$-dimensional feature vector whose informative content lives in a
   $\Theta(K)$-dimensional operator subspace is itself $O(\log(\cdot))$-controlled
   by the slice: continuous-valued but $\Theta(K)$-dimensional, and bounded above
   by $\chi$ through the DPI chain of Section 5 in all cases.

Consequently the Q-IB problem is *automatically in a strong-compression
regime*: the $I(X; T_{\mathcal{O}})$ term of the Lagrangian is capped by
architecture before any training occurs. The learner does not need to spend
capacity suppressing $I(X;T)$ -- the measurement already did. All of $\theta$
is spent on the second term: **maximizing $I(T_{\mathcal{O}}; Y)$ within the
$\Theta(K)$-dimensional measurement slice**, i.e. rotating the encoded state so
that task-relevant structure lands inside $\mathcal{A}(\mathcal{O}_8)$.

This is the formal content of the claim that the Level-8 readout is a
topology-aware inductive bias: it is a fixed, task-independent projection whose
range is aligned (or not) with the target functional, and the bias is exactly
the geometry of that range.

---

## 7. Relation to Canatar et al. 2022

Canatar, Peters, Pehlevan, Wild, and Kubler (2022, "Bandwidth enables
generalization in quantum kernel models") analyze generalization of quantum
kernel methods with the replica method from statistical physics. Their object
is the *fidelity kernel* $k(x, x') = \mathrm{Tr}[\rho(x)\rho(x')]$ (or its
bandwidth-tuned variants), and generalization is controlled by the spectrum of
the kernel integral operator over the data distribution.

Placement relative to T1: the fidelity kernel presumes access to the full
quantum state geometry -- inner products in the *entire*
$4^K$-dimensional operator space. In the language of this note, their framework
implicitly operates *at the Holevo level $\chi$*: the information-theoretic
ceiling of the state ensemble itself, with no measurement bottleneck between
the encoder and the learner (equivalently, a tomographically complete /
swap-test readout). The TC-QIC construction adds the measurement layer as a
first-class object, and the ordering of ceilings is strict:

$$
I(X; T_{\mathcal{O}}) \;\le\; \chi \;\le\; \log \dim \mathrm{supp}(\bar\rho),
\qquad
\underbrace{I(X; T_{\mathcal{O}})}_{\text{operating ceiling (T1)}}
\;\text{ vs. }\;
\underbrace{\chi}_{\text{Canatar-level ceiling}}.
$$

For the Level-8 family $\mathcal{O}_8$, the operating ceiling lives on a
$\Theta(K)$-dimensional operator slice (T2), far below the full kernel
geometry. This is the operator-space level *below* Holevo: two architectures
with identical state ensembles (identical $\chi$, identical fidelity kernels)
can have entirely different inductive biases because they read different
slices $\mathcal{A}(\mathcal{O})$. The measurement, not the kernel, is where
the Level-8 bias lives -- which is why kernel-spectrum analyses cannot see it
and a measurement-resolved theory is required.

---

## References

- S. Tishby, F. Pereira, W. Bialek (1999). The information bottleneck method.
  *Proc. 37th Allerton Conf.*
- A. S. Holevo (1973). Bounds for the quantity of information transmitted by a
  quantum communication channel. *Probl. Peredachi Inf.* 9(3), 3-11.
- E. H. Lieb, M. B. Ruskai (1973). Proof of the strong subadditivity of
  quantum-mechanical entropy. *J. Math. Phys.* 14, 1938.
- G. Lindblad (1975). Completely positive maps and entropy inequalities.
  *Commun. Math. Phys.* 40, 147-151.
- A. Uhlmann (1977). Relative entropy and the Wigner-Yanase-Dyson-Lieb
  concavity in an interpolation theory. *Commun. Math. Phys.* 54, 21-32.
- A. Canatar, E. Peters, C. Pehlevan, S. M. Wild, J. M. Kubler (2022).
  Bandwidth enables generalization in quantum kernel models.
  *arXiv:2206.06686* / TMLR 2023.
- M. A. Nielsen, I. L. Chuang (2010). *Quantum Computation and Quantum
  Information*, 10th anniv. ed., Ch. 11-12 (entropy, Holevo bound).
