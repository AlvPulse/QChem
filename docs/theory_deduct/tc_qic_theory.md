# Topology-Conditioned Quantum Information Compression (TC-QIC): A First-Principles Theory of Quantum Inductive Bias

*Theoretical backbone. All empirical anchors trace to `docs/05`, `docs/06`, `docs/09` and the propositions therein.*

---

## Abstract

We present a first-principles theoretical framework for Topology-Conditioned Quantum Information Compression (TC-QIC), a quantum machine learning architecture for molecular property prediction. Starting from the classical Information Bottleneck (IB), we derive the Quantum IB Lagrangian and show that a fixed measurement structure acts as a hard restriction of the channel feasibility set -- the formal definition of an inductive bias. Two interlocking bottlenecks emerge: (i) the **topological bottleneck**, a spectral low-pass projection of molecular graphs onto K macro-clusters, and (ii) the **operator geometry bottleneck**, a projection of the quantum state onto a Theta(K)-dimensional subspace of the 4^K-dimensional operator space. Their conjunction enforces S_K-equivariance, yields polynomial-in-K generalization guarantees, and preserves poly(K) gradient norms via readout locality. The Absorbability Theorem certifies that the structured-vs-scrambled control is non-vacuous exactly when topology enters as fixed per-molecule data upstream of the trainable head. A Master Theorem unifies these properties and maps onto a phase diagram over topology alignment alpha and readout locality kappa. Every theoretical object is anchored to a measured number; eight falsifiable predictions are stated for future experiments.

---

## Notation

| Symbol | Meaning |
|---|---|
| $X$ | Input random variable (molecule) |
| $Y$ | Target random variable (toxicity label) |
| $T$ | Learned representation |
| $H(X)$ | Shannon entropy of $X$ |
| $I(X;Y)$ | Mutual information between $X$ and $Y$ |
| $\beta$ | Lagrange multiplier in IB; compression-relevance trade-off |
| $\mathcal{H}$ | Hilbert space $(\mathbb{C}^2)^{\otimes K}$ |
| $K$ | Number of qubits / macro-clusters |
| $\rho_\theta(x)$ | Density operator produced by encoder circuit with parameters $\theta$ |
| $\mathcal{D}(\mathcal{H})$ | Set of density operators on $\mathcal{H}$ |
| $S(\rho)$ | Von Neumann entropy $-\operatorname{Tr}\rho\log\rho$ |
| $\chi$ | Holevo quantity |
| $I_Q(X;T)$ | Quantum mutual information of the cq-state |
| $\mathcal{O}$ | Set of measurement observables $\{O_a\}$ |
| $\mathcal{M}_{\mathcal{O}}$ | Measurement CPTP channel |
| $\phi_{\mathcal{O}}(\rho)$ | Classical feature vector from readout |
| $\mathcal{A}(\mathcal{O})$ | Accessible operator subspace spanned by $\mathcal{O}$ |
| $\mathrm{Herm}(\mathcal{H})$ | Real vector space of Hermitian operators on $\mathcal{H}$, dim $4^K$ |
| $\langle O, O' \rangle_{\mathrm{HS}}$ | Hilbert-Schmidt inner product $2^{-K}\operatorname{Tr}[O^\dagger O']$ |
| $P_\mu$ | Pauli string indexed by $\mu \in \{I,X,Y,Z\}^K$ |
| $C_{ij}$ | Connected two-qubit correlator $\langle O_i O_j \rangle - \langle O_i \rangle \langle O_j \rangle$ |
| $A$ | Adjacency / coarse-grained bond-weight matrix |
| $B_A(\rho)_i$ | Bond-pooled readout $\sum_j A_{ij} C_{ij}(\rho)$ |
| $G=(V,E,w)$ | Molecular graph with vertex set $V$, edge set $E$, weight $w$ |
| $L$ | Graph Laplacian $D - W$ |
| $(\lambda_k, u_k)$ | Eigenpairs of $L$, $0=\lambda_1 \le \dots \le \lambda_n$ |
| $C(G) = G_K$ | Coarse-grained graph with $K$ clusters |
| $\mathcal{S}(K)$ | Placed-signal subspace $\operatorname{span}\{C_{ij}: A_{ij}>0\}$ |
| $\eta_{\mathcal{O}}(K)$ | Signal-readout alignment ratio |
| $\alpha$ | Topology alignment $\langle A_{\mathrm{read}}, A_{\mathrm{place}} \rangle / (\|A_{\mathrm{read}}\| \|A_{\mathrm{place}}\|)$ |
| $\kappa$ | Readout locality (body order of observables) |
| $n$ | Number of training samples |
| $\hat{\mathfrak{R}}_n$ | Empirical Rademacher complexity |
| $R(h)$ | Population risk of hypothesis $h$ |
| $\hat{R}_n(h)$ | Empirical risk of hypothesis $h$ |
| $\epsilon$ | Coarse-graining information loss $I(G;Y) - I(C(G);Y)$ |
| $\Delta$AUC | Structured minus scrambled AUC difference |
| $n^\star$ | Classical-crossover sample size |
| $K^\star$ | Predicted saturation qubit count for K-scaling law |

---

## Section 1 -- The Quantum Information Bottleneck from First Principles

### 1.1 Classical information-theoretic primitives

Let $X$ be the input random variable (a molecule), $Y$ the target (toxicity label), and $T$ a learned representation. We use Shannon entropy $H(X) = -\sum_x p(x)\log p(x)$, conditional entropy $H(X \mid Y)$, and **mutual information**

$$
I(X;Y) \;=\; H(X) - H(X \mid Y) \;=\; \sum_{x,y} p(x,y)\,\log\frac{p(x,y)}{p(x)\,p(y)} \;\ge 0 .
\tag{1.1}
$$

A representation obeys the Markov chain $Y \to X \to T$ (the label acts on the representation only through the input). The **Data Processing Inequality (DPI)** then gives $I(T;Y) \le I(X;Y)$: post-processing cannot create task information.

**Definition 1.1 (Classical Information Bottleneck, Tishby-Pereira-Bialek).** The IB representation solves

$$
\min_{p(t \mid x)}\;\; \mathcal{L}_{\mathrm{IB}}[p(t \mid x)] \;=\; I(T;X) \;-\; \beta\, I(T;Y), \qquad \beta > 0 .
\tag{1.2}
$$

The first term is a **compression** pressure (make $T$ forget $X$); the second is a **relevance** pressure (make $T$ predict $Y$). The stationary solution satisfies the self-consistent equations

$$
p(t \mid x) \propto p(t)\exp\!\big[-\beta\, D_{\mathrm{KL}}\!\big(p(y \mid x) \,\|\, p(y \mid t)\big)\big].
\tag{1.3}
$$

The Lagrange multiplier $\beta$ traces the optimal compression-relevance frontier.

The key structural fact for what follows: in the classical IB, the *channel* $p(t \mid x)$ is **free** -- the optimizer ranges over all stochastic maps. An inductive bias is, precisely, a **restriction of that feasible set**.

### 1.2 The quantum encoder and its mutual information

A parametrized quantum circuit is an encoder that maps each input to a density operator on $\mathcal{H} = (\mathbb{C}^2)^{\otimes K}$:

$$
\mathcal{E}_\theta:\; x \;\longmapsto\; \rho_\theta(x) \in \mathcal{D}(\mathcal{H}), \qquad \rho_\theta(x) = U_\theta(x)\,|0\rangle\!\langle 0|\,U_\theta(x)^\dagger .
\tag{1.4}
$$

To speak of $I_Q(T;X)$ we form the **classical-quantum (cq) state**

$$
\rho_{XT} \;=\; \sum_x p(x)\, |x\rangle\!\langle x|_X \otimes \rho_\theta(x)_T .
\tag{1.5}
$$

**Definition 1.2 (Quantum mutual information of the encoder).** With von Neumann entropy $S(\rho) = -\operatorname{Tr}\rho\log\rho$,

$$
I_Q(X;T)_{\rho_{XT}} \;=\; S(\rho_X) + S(\rho_T) - S(\rho_{XT})
\;=\; \underbrace{S\!\Big(\textstyle\sum_x p(x)\rho_\theta(x)\Big) - \sum_x p(x)\,S\big(\rho_\theta(x)\big)}_{\displaystyle \chi(\{p(x),\rho_\theta(x)\})}.
\tag{1.6}
$$

For a cq-state the quantum mutual information equals the **Holevo quantity** $\chi$. This is the pre-measurement information the quantum representation holds about $X$.

### 1.3 The measurement bottleneck as a CPTP channel

We never access $\rho_\theta(x)$ directly; we read a fixed set of observables $\mathcal{O} = \{O_a\}_{a=1}^{d}$ and form the **classical feature vector**

$$
\phi_{\mathcal{O}}(\rho) \;=\; \big(\operatorname{Tr}[O_a\,\rho]\big)_{a=1}^{d} \in \mathbb{R}^d .
\tag{1.7}
$$

Reading a fixed observable family is a completely-positive trace-preserving (CPTP) measurement channel $\mathcal{M}_{\mathcal{O}}$ composed with a classical embedding. It is **entanglement-breaking**: $\mathcal{M}_{\mathcal{O}}(\rho) = \sum_a \operatorname{Tr}[O_a\rho]\,|a\rangle\!\langle a|$.

**Lemma 1.1 (Measurement is a bottleneck).** Let $T_{\mathcal{O}} = \mathcal{M}_{\mathcal{O}}(\rho_\theta(x))$. Because $\mathcal{M}_{\mathcal{O}}$ is CPTP, the quantum DPI gives

$$
I\big(X;T_{\mathcal{O}}\big) \;\le\; I_Q(X;T) \;=\; \chi\big(\{p(x),\rho_\theta(x)\}\big) .
\tag{1.8}
$$

*Proof.* Quantum mutual information is monotone non-increasing under CPTP maps on either subsystem; apply $\mathrm{id}_X \otimes \mathcal{M}_{\mathcal{O}}$ to $\rho_{XT}$. $\square$

The accessible information is further bounded by Holevo's theorem, $I_{\mathrm{acc}} \le \chi$. Thus **which observables we read strictly sets the ceiling on extractable information.**

### 1.4 Operator geometry: the accessible subspace

The space of observables on $K$ qubits is the real vector space $\mathrm{Herm}(\mathcal{H})$ of dimension $4^K$, with the **Hilbert-Schmidt inner product** $\langle O, O' \rangle_{\mathrm{HS}} = 2^{-K}\operatorname{Tr}[O^\dagger O']$. The Pauli strings $\{P_\mu\}_{\mu \in \{I,X,Y,Z\}^K}$ form an orthonormal basis.

**Definition 1.3 (Accessible operator subspace).** The readout $\mathcal{O}$ spans a subspace $\mathcal{A}(\mathcal{O}) = \operatorname{span}\{O_a\} \subseteq \mathrm{Herm}(\mathcal{H})$. The feature map $\phi_{\mathcal{O}}$ depends on $\rho$ **only** through its orthogonal projection $\Pi_{\mathcal{A}}\rho$ onto $\mathcal{A}(\mathcal{O})$; the complement $\mathcal{A}^\perp$ is invisible.

For the Level-8 readout $\mathcal{O}_8 = \{X_i, Y_i, Z_i\}_i \cup \{\sum_j A_{ij}Z_iZ_j,\ \sum_j A_{ij}X_iX_j\}_i$, the accessible subspace consists of

- $3K$ weight-1 Paulis, and
- $A$-weighted combinations of weight-2 Paulis $Z_iZ_j, X_iX_j$ supported on bonded pairs.

Hence, for a sparse molecular graph with $|E| = \Theta(K)$ bonds,

$$
\boxed{\;\dim \mathcal{A}(\mathcal{O}_8) \;=\; \Theta(K) \quad\text{out of}\quad \dim\mathrm{Herm}(\mathcal{H}) = 4^K .\;}
\tag{1.9}
$$

This is the **operator geometry bottleneck** made precise: the readout projects the state onto a $\Theta(K)$-dimensional slice of a $4^K$-dimensional operator space.

### 1.5 Trace-out is an inductive bias, not noise

The unobserved directions $\mathcal{A}^\perp$ are not stochastic corruption; they are **deterministically discarded** by architecture. Decompose any observable-linear predictor's target functional as

$$
f(\rho) = \langle W, \Pi_{\mathcal{A}}\rho \rangle_{\mathrm{HS}} + \langle W, \Pi_{\mathcal{A}^\perp}\rho \rangle_{\mathrm{HS}} .
\tag{1.10}
$$

The readout forces the second term to zero.

**Proposition 1.2 (Trace-out as structured prior).** Let $g^\star(\rho) = \mathbb{E}[Y \mid \rho]$ be the Bayes predictor and write $g^\star = g^\star_{\mathcal{A}} + g^\star_{\mathcal{A}^\perp}$ by the HS decomposition. Restricting to $\mathcal{O}$:

1. is **lossless (zero-cost prior)** iff $g^\star_{\mathcal{A}^\perp} = 0$, i.e. the label is independent of the traced-out operators;
2. is **lossy but variance-reducing** otherwise, contributing an approximation error $\|g^\star_{\mathcal{A}^\perp}\|$ while removing $\dim\mathcal{A}^\perp$ degrees of freedom from the estimation problem.

Either way it is a **hard constraint on the hypothesis class**, identical in role to translation-equivariance in a CNN -- a bias, not noise. This is the formal content of the "operator geometry bottleneck" that the phenomenological Q-TIB only asserted.

### 1.6 The Quantum Information Bottleneck Lagrangian

**Definition 1.4 (Q-IB Lagrangian).** With the measured representation $T_{\mathcal{O}} = \mathcal{M}_{\mathcal{O}}(\rho_\theta(x))$,

$$
\mathcal{L}_{\mathrm{Q\text{-}IB}}(\theta, \mathcal{O}) \;=\; I\!\big(X;T_{\mathcal{O}}\big) \;-\; \beta\, I\!\big(T_{\mathcal{O}};Y\big) .
\tag{1.11}
$$

The decisive difference from Definition 1.1 is the optimization domain. The classical IB minimizes over *all* channels $p(t \mid x)$. The Q-IB minimizes over **circuit parameters $\theta$ at fixed measurement structure $\mathcal{O}$**:

$$
\min_{\theta}\;\mathcal{L}_{\mathrm{Q\text{-}IB}}(\theta, \mathcal{O}), \qquad \mathcal{O}\ \text{fixed by architecture.}
\tag{1.12}
$$

**The choice of $\mathcal{O}$ is the inductive bias.** By Lemma 1.1 the compression term is upper-bounded, $I(X;T_{\mathcal{O}}) \le \Theta(K) \cdot \log(\cdot)$ (at most $\dim\mathcal{A}$ real coordinates of information), so the Q-IB is *automatically* in a strong-compression regime. The learner spends its freedom $\theta$ maximizing $I(T_{\mathcal{O}};Y)$ *within* the $\Theta(K)$-dimensional topology-aligned slice. TC-QIC is the special case $\mathcal{O} = \mathcal{O}_8$: the accessible slice is the *graph-weighted correlator subspace*.

---

## Section 2 -- Generalization Theory of TC-QIC

*Foundations: The generalization theory in this section takes as its parent framework the kernel-machine learning-curve analysis of Canatar, Bordelon, and Pehlevan (2022). Section 2 extends rather than replaces their work: Canatar et al. study kernel machines with a scalar bandwidth parameter and derive exact generalization curves via the replica method; here we study variational quantum circuits whose effective kernel carries a per-molecule, graph-structured bandwidth encoded in the adjacency matrix A. Their replica-method formula (Eq. 9 of their paper) provides a more precise characterization of the bias-variance trade-off in the kernel view than the Rademacher bound of Theorem 2.2, which applies to the linear-head regime and is tight only in the worst case over distributions. Deriving the TC-QIC analog of their Eq. 9 -- replacing the scalar bandwidth with the spectral structure of A -- is left as an important direction for future work.*

### 2.1 The hypothesis class

**Definition 2.1.** The TC-QIC hypothesis class with readout $\mathcal{O}$, encoder parameters $\Theta$, and linear head is

$$
\mathcal{H}_{\mathcal{O},\Theta} = \Big\{\,x \mapsto w^\top\phi_{\mathcal{O}}\big(\rho_\theta(x)\big) \;:\; \theta \in \Theta,\ w \in \mathbb{R}^{d},\ \|w\|_2 \le W\,\Big\}, \qquad d = \dim\mathcal{A}(\mathcal{O}) .
\tag{2.1}
$$

For fixed $\theta$ this is a linear class over a $d$-dimensional feature map. The Level-8 head has $d = 3K + 2K = 5K$ features (single-qubit $X,Y,Z$ plus two bond-pooled channels), so $d = \Theta(K)$.

### 2.2 Rademacher complexity and the generalization bound

**Lemma 2.1 (Bounded features).** Each Pauli expectation lies in $[-1,1]$. Because $A$ is max-normalized ($\max_{ij}A_{ij} \le 1$) and molecular clusters have bounded degree $\bar d = O(1)$, each bond-pooled feature satisfies $|b[i]| = |\sum_j A_{ij}\langle O_iO_j\rangle| \le \bar d$. Hence

$$
\|\phi_{\mathcal{O}_8}(\rho)\|_2 \;\le\; \sqrt{3K + 2K\bar d^{\,2}} \;=\; \Theta(\sqrt{K}) \;=:\; B .
\tag{2.2}
$$

**Theorem 2.2 (Generalization bound for TC-QIC).** For $n$ i.i.d. samples, the empirical Rademacher complexity of the linear head over the bounded feature map obeys $\hat{\mathfrak{R}}_n(\mathcal{H}_{\mathcal{O}_8}) \le BW/\sqrt{n} = O\!\big(W\sqrt{K/n}\big)$. Consequently, for any $1$-Lipschitz loss, with probability $\ge 1-\delta$ over the sample,

$$
\boxed{\;R(h) \;\le\; \hat R_n(h) \;+\; 2\,O\!\Big(W\sqrt{\tfrac{K}{n}}\Big) \;+\; 3\sqrt{\frac{\ln(2/\delta)}{2n}} \;.}
\tag{2.3}
$$

*Proof.* Standard linear-class Rademacher bound $\hat{\mathfrak R}_n \le \max_i\|\phi(x_i)\|_2\,W/\sqrt n$ (Bartlett-Mendelson), plus Lemma 2.1, plus the Rademacher generalization theorem. $\square$

**Corollary 2.3 (Exponential sample-complexity saving).** A full-Hilbert readout ($\mathcal{O} =$ all $4^K$ Paulis) has $B = \Theta(2^K)$ in the worst case, giving $\hat{\mathfrak R}_n = O(2^K W/\sqrt n)$ and demanding $n = \Omega(4^K)$ samples for a fixed gap. The operator geometry bottleneck reduces the required sample size from $\Omega(4^K)$ to $O(K)$: the generalization gap grows as $\sqrt{K}$, not $2^K$. **This is why TC-QIC is trainable and generalizes where an unrestricted quantum readout cannot.**

### 2.3 The bond-pooled readout as a symmetry constraint

Let $\pi \in S_K$ permute qubits/clusters, $U_\pi$ the corresponding unitary, and $A^\pi$ the relabeled adjacency, $A^\pi_{ij} = A_{\pi^{-1}(i)\,\pi^{-1}(j)}$.

**Theorem 2.4 (Permutation equivariance of the readout).** The bond-pooled readout is $S_K$-equivariant:

$$
B_{A^\pi}\!\big(U_\pi\rho\,U_\pi^\dagger\big)_i \;=\; B_{A}(\rho)_{\pi^{-1}(i)} .
\tag{2.4}
$$

*Proof.* $B_A(\rho)_i = \sum_j A_{ij}\operatorname{Tr}[O_iO_j\rho]$. Under $\rho \mapsto U_\pi\rho U_\pi^\dagger$ the correlator $\operatorname{Tr}[O_iO_j\,U_\pi\rho U_\pi^\dagger] = \operatorname{Tr}[O_{\pi^{-1}(i)}O_{\pi^{-1}(j)}\rho]$, and $A^\pi_{ij} = A_{\pi^{-1}(i)\pi^{-1}(j)}$. Reindexing $k = \pi^{-1}(i),\,l = \pi^{-1}(j)$ gives $\sum_l A_{kl}\operatorname{Tr}[O_kO_l\rho] = B_A(\rho)_k$. $\square$

**Corollary 2.5 (Equivariant hypothesis class).** $\mathcal{H}_{\mathcal{O}_8} \subseteq \mathcal{F}_{\mathrm{equiv}}(G)$, the class of functions equivariant under the molecular-graph automorphism group $\mathrm{Aut}(G) \le S_K$. Restricting to $\mathcal{F}_{\mathrm{equiv}}$ collapses parameters along group orbits: the effective number of free directions is reduced by $\sim |\mathrm{Aut}(G)|$, tightening $\hat{\mathfrak R}_n$ by the corresponding orbit-averaging factor. The bond-pooled readout therefore acts as an *automatic equivariance regularizer* -- the same mechanism that makes GNNs generalize, obtained here from measurement geometry rather than weight tying.

### 2.4 Formal analogy to classical GNN message passing

A message-passing GNN layer computes $h_i' = \varphi\big(h_i,\ \bigoplus_{j \in N(i)}\psi(h_i,h_j,A_{ij})\big)$; its inductive bias is the aggregation $\bigoplus_{j \in N(i)}$ over graph neighborhoods.

**Proposition 2.6 (Quantum message passing).** The Level-8 readout $b[i] = \sum_j A_{ij}\,C_{ij}(x)$ is exactly one message-passing step with

$$
\psi(i,j) \;=\; A_{ij}\,C_{ij}(x) \;=\; A_{ij}\big(\langle O_iO_j\rangle - \langle O_i\rangle\langle O_j\rangle + \langle O_i\rangle\langle O_j\rangle\big), \qquad \bigoplus = \textstyle\sum_{j} .
\tag{2.5}
$$

The message is a **genuine two-qubit connected correlator** $C_{ij}$ -- a non-classical quantity that can carry entanglement -- rather than a learned function of node features. The two inductive biases are formally identical at the level of the aggregation operator (both restrict $\mathcal{H}$ to graph-local sum-pooled functions); they differ only in the message alphabet. This is the precise sense in which "Level 8 is a quantum analogue of GNN neighborhood aggregation."

### 2.5 The Absorbability Theorem as function-class equivalence

We restate the project's central methodological result as a theorem about equality of hypothesis classes under reparametrization.

**Theorem 2.7 (Absorbability = function-class identity; Proposition 1).** Let each encoding site $s$ apply a gate with angle $\theta_s = \langle w_s, h \rangle$ for a free trainable row $w_s$ and shared feature $h = f(\mathrm{mol})$. Let the scramble fix a permutation $\pi_s$ acting once on the coordinates driving site $s$. If every site reads through its own free projection under a single permutation, then

$$
\mathcal{H}_{\mathrm{struct}} \;=\; \mathcal{H}_{\mathrm{scram}} \qquad\text{(equality of function classes).}
\tag{2.6}
$$

*Proof.* The site angle vector is $\theta = Wh$ with $W$ unconstrained; a fixed output permutation is $PW$. Setting $W' = P^{-1}W_{\mathrm{struct}}$ yields $PW' = W_{\mathrm{struct}}$, so the scrambled site computes identical angles. Independence across sites makes the assignment simultaneously satisfiable; the reparametrization $\theta \mapsto \theta'$ is a bijection of parameter space intertwining the two families. $\square$

**Corollary 2.8 (When the risk gap is meaningful; Corollary 2).** Since $\mathcal{H}_{\mathrm{struct}} = \mathcal{H}_{\mathrm{scram}}$ implies $\inf_{h \in \mathcal{H}_{\mathrm{struct}}}R(h) = \inf_{h \in \mathcal{H}_{\mathrm{scram}}}R(h)$, the population risk gap is **exactly zero** and any measured gap is optimization/finite-sample noise. Non-absorbability -- hence a genuine gap -- holds iff **(a)** one shared $h$ is routed under $\ge 2$ inconsistent permutations, or **(b)** the structured quantity enters as **fixed per-molecule data** $A(\mathrm{mol})$ multiplying a physical observable **upstream of the only trainable layer** (no weight to permute; $A$ varies per molecule so no single $W'$ maps scrambled to structured for all molecules at once). **Level 8 satisfies (b).** This converts "the control failed" into a checkable pre-registration criterion: trace each structured signal to the trainable map in front of it and verify (a) or (b).

---

## Section 3 -- Chemical Topology as Structural Prior

### 3.1 Formal molecular graph and its symmetry

**Definition 3.1.** A molecule is a weighted graph $G = (V, E, w)$, $|V| = n$ atoms, edge weights $w: E \to \mathbb{R}_+$ (bond orders), with node attributes $f: V \to \mathbb{R}^5$ (atomic number, Gasteiger charge, degree, aromaticity, in-ring). The symmetric group $S_n$ acts by relabeling atoms, $\sigma \cdot G$.

**Axiom 3.2 (Label invariance).** Toxicity is a molecular property: $Y(\sigma \cdot G) = Y(G)$ for all $\sigma \in S_n$. Equivalently $Y$ factors through the isomorphism class $[G] \in \mathcal{G}/S_n$.

**Proposition 3.3 (Orbit is a sufficient statistic).** Because $Y$ is $S_n$-invariant, the quotient map $q: G \mapsto [G]$ is a sufficient statistic for $Y$: every admissible predictor factors as $\hat Y = g \circ q$. Any architecture that is **not** invariant under $S_n$ wastes capacity distinguishing relabelings that carry no signal. This is the first-principles reason a chemistry ML model *must* be permutation-equivariant -- and, via Corollary 2.5, why TC-QIC's equivariant readout is aligned with the task symmetry.

### 3.2 Coarse-graining as spectral low-pass filtering

**Definition 3.4 (Coarse-graining functor).** Let $L = D - W$ be the graph Laplacian with eigenpairs $(\lambda_k, u_k)$, $0 = \lambda_1 \le \dots \le \lambda_n$. Spectral clustering partitions $V$ into $K$ clusters via the bottom-$K$ eigenvectors, producing $C: G \mapsto G_K = (V_K, A, q)$ with $A_{cc'} = \sum_{i \in c, j \in c'}w_{ij}$ (normalized) and $q_c = |c|^{-1}\sum_{i \in c}f_i$.

**Lemma 3.5 (Coarse-graining is an ideal low-pass filter).** The ratio-cut relaxation solved by spectral clustering projects node signals onto $\operatorname{span}(u_1, \dots, u_K)$, the smallest-$\lambda$ (lowest graph-frequency) eigenspace. Writing a node signal in the graph Fourier basis $\hat f_k = \langle u_k, f \rangle$, coarse-graining retains $\{\hat f_k: k \le K\}$ and discards $\{\hat f_k: k > K\}$.

**Corollary 3.6 ("High-frequency atomic noise" is defined).** The phrase from the phenomenological Q-TIB acquires a precise meaning: the discarded content is exactly the high-graph-frequency ($\lambda_k$ large, $k > K$) component of the node-feature field. The coarse-graining is the *ideal low-pass projector* $\Pi_{\le K} = \sum_{k \le K}u_ku_k^\top$ onto macro-topology, removing rapidly-varying local variance *before* it enters the circuit. This is the **topological bottleneck**, now a genuine spectral operation rather than an informal claim.

### 3.3 Sufficiency and the failure mode of coarse-graining

**Definition 3.7 (epsilon-sufficiency).** $C(G)$ is $\epsilon$-sufficient for $Y$ if the residual within-cluster information is small:

$$
I\big(G;Y \mid C(G)\big) \;=\; I(G;Y) - I\big(C(G);Y\big) \;\le\; \epsilon .
\tag{3.1}
$$

**Theorem 3.8 (Conditional sufficiency of macro-topology).** If every label-relevant substructure (toxicophore, aromatic ring, functional group) is contained within a single cluster of the partition -- i.e. clustering does not split a toxicophore across the cut -- then $I(G;Y \mid C(G)) \approx 0$ and $C(G)$ is $\epsilon$-sufficient with $\epsilon \to 0$. Conversely, if a toxicophore straddles a cluster boundary, its intra-fragment signal falls in the discarded high-frequency band (Lemma 3.5) and $\epsilon$ is bounded below by that fragment's contribution.

This is the formal statement of **Phase A** ("Information Retained vs. Downstream AUC") and simultaneously the first **boundary condition**: the topological prior is justified exactly when the coarse-graining respects chemical substructure, and misspecified otherwise. It also explains the empirical AUC ceiling of approximately $0.61$-$0.66$: coarse-graining to $K \in \{4,6,8\}$ clusters on molecules of $20$-$40$ heavy atoms forces $\epsilon > 0$, capping absolute performance regardless of the circuit.

### 3.4 The bond-pooled readout is a minimal sufficient statistic

**Theorem 3.9 (Minimal sufficiency of $B_A$ for topology-conditioned inference).** Consider the family of readouts that are (i) at most $2$-local, (ii) linear in the two-qubit correlators, and (iii) $S_K$-equivariant w.r.t. graph automorphisms. Suppose the graph-gated entangler places the label-relevant signal in the connected correlators $C_{ij}$ on bonded pairs (Section 4). Then the bond-pooled statistic

$$
B_A(\rho)_i \;=\; \sum_j A_{ij}\,C_{ij}(\rho)
\tag{3.2}
$$

is the **minimal sufficient statistic** within this family for inferring the topology-conditioned target.

*Proof sketch.* Sufficiency: the correlator tensor $\{C_{ij}\}$ carries all pairwise information; contracting it with $A$ retains precisely the $A$-aligned projection, which by hypothesis is where the signal lives, so no relevant information is lost (Fisher-Neyman factorization through the $A$-weighted contraction). Minimality: any coarser statistic drops connected correlators and is blind to the signal (Lemma 3.10, below); any finer statistic (the full $\{C_{ij}\}$) carries the off-bond correlators, which are empirically near-zero ($|C_{ij}| = 0.013$ off-bond vs $0.066$ on-bond, section C.4) and hence pure nuisance under the equivariance constraint. Thus $B_A$ is the unique reduction retaining signal while remaining equivariant. $\square$

**Lemma 3.10 (Single-qubit readout is blind to the graph signal; Lemma 3).** The single-qubit readout $S: \rho \mapsto (\langle X_i \rangle, \langle Y_i \rangle, \langle Z_i \rangle)_i$ is a function of the $K$ one-qubit marginals $\{\rho_i\}$ only; two states with identical marginals but different correlations are indistinguishable to $S$. The graph-gated $\mathrm{IsingXX}(A_{ij}\theta)$ writes topology into the connected correlators $C_{ij} = \langle P_iP_j \rangle - \langle P_i \rangle\langle P_j \rangle$, which are by definition not determined by marginals. There are $\Theta(K^2)$ pairwise correlators but only $3K = \Theta(K)$ single-qubit terms; for a sparse graph the signal lives in $\Theta(K)$ on-bond correlators that $B_A$ reads exactly and $S$ cannot.

### 3.5 The K-scaling law, derived

Define the **signal-readout alignment**

$$
\eta_{\mathcal{O}}(K) = \frac{\dim\!\big(\mathcal{A}(\mathcal{O}) \cap \mathcal{S}(K)\big)}{\dim\mathcal{S}(K)},
\tag{3.3}
$$

where $\mathcal{S}(K) = \operatorname{span}\{C_{ij}: A_{ij} > 0\}$ is the placed-signal subspace, of dimension $s(K) = |E| = \tfrac{1}{2}\bar d\,K = \Theta(K)$.

- **Level 8 ($B_A$):** $\mathcal{S}(K) \subseteq \mathcal{A}(\mathcal{O}_8)$, so $\eta_{B}(K) = 1$ for all $K$.
- **Gate-only ($S$):** $\mathcal{S}(K) \cap \mathcal{A}(S) = \{0\}$ (marginals contain no connected correlator), so the accessible share of the signal is $\eta_S(K) = \Theta(1/K) \to 0$.

**Proposition 3.11 (Bias scaling law).** Model the detectable effect as (aligned signal mass) minus (generalization penalty from Theorem 2.2):

$$
\Delta(K) \;\propto\; \eta_{\mathcal{O}}(K) \cdot s(K) \;-\; c\,W\sqrt{K/n} .
\tag{3.4}
$$

For Level 8, $\eta_B s(K) = \Theta(K)$: the bias **grows linearly** in $K$ until the generalization penalty or the coarse-graining information ceiling (Theorem 3.8) causes saturation. For gate-only, $\eta_S(K)s(K) = \Theta(1/K) \cdot \Theta(K) = \Theta(1)$ -- flat aligned mass, so once the $\sqrt{K/n}$ penalty and the noise floor rise, the *detectable* effect **fades below significance**.

**Least-squares realization on the data** ($K$, median $\Delta$AUC) $=$ (4, 0.0078), (6, 0.0108), (8, 0.0134):

$$
\boxed{\;\Delta_B(K) \;\approx\; 1.4 \times 10^{-3}\,K \;+\; 2.3 \times 10^{-3}\;} \qquad (R^2 \approx 0.996),
\tag{3.5}
$$

a near-perfect linear law with slope $\approx +0.0014$ AUC/qubit -- exactly the $\Theta(K)$ growth Proposition 3.11 predicts. Gate-only sits at a flat $\approx 3 \times 10^{-3}$, below the $80\%$-power minimum-detectable effect $6.6 \times 10^{-3}$ (section V.1), so its fade is *predicted*, not merely observed.

**Saturation prediction.** Growth halts at $K^\star$ where the coarse-graining exhausts macro-topology, $\partial_K I(C(G);Y) \to 0$, i.e. $K^\star \approx$ (heavy atoms)/(atoms per chemically-coherent fragment). For Tox21 ($\sim 20$-$40$ heavy atoms, fragments of $\sim 3$-$5$ atoms), $K^\star \approx 8$-$16$: the linear law should continue through $K \approx 10$-$12$ and then bend over.

---

## Section 4 -- The TC-QIC Master Theorem and Phase Diagram

### 4.1 The place-then-harvest operator identity

**Lemma 4.1 (Placement).** The graph-gated layer $U(A, \theta) = \prod_{i<j}\exp\!\big[-\tfrac{i}{2}A_{ij}\theta_{ij}X_iX_j\big]$ generates, to leading order in $\theta$, connected correlators $C_{ij} = \Theta(A_{ij}\theta_{ij})$ concentrated on bonded pairs ($A_{ij} > 0$) and vanishing on non-bonded pairs ($A_{ij} = 0$). The entangler cannot create correlation where $A_{ij} = 0$; it can only scale it where $A_{ij} > 0$.

**Lemma 4.2 (Harvest).** $B_A$ contracts the correlator tensor against the *same* $A$, so it reads out precisely the subspace on which Lemma 4.1 placed correlation. With a mismatched adjacency $A' \ne A$, the harvest picks up $\sum_j A'_{ij}C_{ij}$, whose overlap with the placed signal is $\langle A', A \rangle / (\|A'\|\|A\|)$ -- maximized at $A' = A$ (structured), $\approx 0$ for random $A'$ (scrambled), and *anti-aligned* if the placement graph and pooling graph differ systematically (meas-only).

### 4.2 The Master Theorem

**Theorem 4.3 (TC-QIC Master Theorem).** Let a $K$-qubit circuit have a graph-gated entangler $U(A, \theta)$ and bond-pooled readout $B_A$, trained on $S_n$-invariant labels of molecules coarse-grained to $A = C(G)$, with $O(1)$ depth and $O(K)$ two-qubit gates. Then:

**(i) Tight, topology-aligned compression.** The measured representation $T = (S(\rho), B_A(\rho))$ realizes $I(X;T) \le \dim\mathcal{A}(\mathcal{O}_8) \cdot O(1) = \Theta(K)$ nats (Lemma 1.1, Definition 1.3), with the retained information concentrated on the placed-signal subspace $\mathcal{S}(K)$ (Lemmas 4.1-4.2). Any topology-agnostic readout of equal output dimension attains strictly smaller relevance $I(T;Y)$ whenever $Y$ depends on graph structure (Lemma 3.10 + DPI).

**(ii) Symmetry.** $T$ is $S_K$-equivariant (Theorem 2.4), so $\mathcal{H}_{TC\text{-}QIC} \subseteq \mathcal{F}_{\mathrm{equiv}}(G)$ and the generalization gap is $O(W\sqrt{K/n})$ (Theorem 2.2), polynomial in $K$.

**(iii) Trainability (no barren plateau in the $O(K)$ regime).** Every observable in $B_A$ is $\le 2$-local and the circuit is shallow; by the local-cost-function theorem (Cerezo et al.), the gradient variance obeys

$$
\operatorname{Var}_\theta\big[\partial_{\theta}\, \mathcal{C}\big] \;=\; \Omega\!\big(\mathrm{poly}(1/K)\big),
\tag{4.1}
$$

not the $\Theta(2^{-K})$ collapse of global-observable deep circuits (McClean et al.). Hence gradients remain estimable and training does not stall as $K$ grows.

*Proof.* (i) combines the operator-subspace dimension count (Definition 1.3), the measurement DPI (Lemma 1.1), and place-then-harvest (Lemmas 4.1-4.2); the strict inequality is Lemma 3.10. (ii) is Theorems 2.4 and 2.2. (iii) invokes the theorem that shallow circuits with local cost observables have at-most-polynomially-vanishing gradient variance; $B_A$ is a sum of $O(K)$ local terms and depth is $O(1)$. $\square$

**Remark (correction to the phenomenological claim).** The old Q-TIB said the circuit is "structurally forbidden from accessing the full Hilbert space." Precisely: the *unitary* still explores $2^K$ dimensions, but the *measurement* projects onto a $\Theta(K)$-dimensional operator slice (Definition 1.3). The barren-plateau resistance in (iii) comes from readout locality + shallow depth, **not** from any restriction on the evolution -- resolving the imprecision flagged in the original framing.

### 4.3 When TC-QIC beats classical IB, and when not

**Theorem 4.4 (Strict-improvement condition).** Let $\mathcal H_{\mathrm{cl}}$ be a classical IB representation of matched output dimension $d = \Theta(K)$ on the same coarse features $q$ (no correlators). TC-QIC attains strictly lower risk than $\mathcal H_{\mathrm{cl}}$ iff

1. **(alignment)** $A = C(G)$ is a faithful macro-topology, $\epsilon$-sufficient with small $\epsilon$ (Theorem 3.8), *and*
2. **(non-classical signal)** the label depends on connected two-qubit correlators $C_{ij}$ not expressible as functions of the node marginals -- i.e. $g^\star_{\mathcal S} \ne 0$ in the correlator subspace -- *and*
3. **(capacity match)** the comparison holds output dimension and parameter count fixed, so the only difference is the message alphabet (quantum correlator vs classical node-feature product).

Under (1)-(3), Lemma 3.10 guarantees TC-QIC accesses signal $\mathcal H_{\mathrm{cl}}$ cannot, giving a strict gap. This is the theoretical statement of the structured > scrambled result and of the (still-open) *param-matched* classical experiment.

**Theorem 4.5 (Classical-dominance condition).** TC-QIC does **not** beat an *unconstrained* classical model whenever

$$
\underbrace{\|g^\star_{\mathcal A^\perp}\|^2}_{\text{TC-QIC approx. error}} \;+\; \underbrace{\epsilon}_{\text{coarse-graining loss}} \;>\; \underbrace{O\!\big(W_{\mathrm{cl}}\sqrt{\mathrm{cap}_{\mathrm{cl}}/n}\big)}_{\text{classical estimation error}} ,
\tag{4.2}
$$

i.e. when there is enough data $n$ that the flexible classical model's estimation error falls below TC-QIC's (approximation error + coarse-graining loss). Because the classical MLP has $\sim 10\times$ parameters and reads $[q \,\|\, \mathrm{vec}(A)]$ directly, its approximation error is much smaller; with $n \approx 7823$ (Tox21) the inequality holds, and classical leads. **This derives the empirical 5-8-point classical lead from the bias-variance decomposition rather than positing it.** It is exactly the "power of data" phenomenon (Huang et al. 2021) specialized to this architecture: the topology bottleneck buys low variance at the price of high approximation error, and at this coarse-graining resolution the price exceeds the discount.

### 4.4 The phase diagram

Two order parameters govern the regime:

$$
\alpha = \frac{\langle A_{\mathrm{read}}, A_{\mathrm{place}} \rangle}{\|A_{\mathrm{read}}\|\,\|A_{\mathrm{place}}\|} \in [-1,1] \quad (\text{topology alignment}), \qquad
\kappa \in \{1, 2, \dots, K\} \quad (\text{readout locality}).
\tag{4.3}
$$

| Region | $(\alpha, \kappa)$ | Behavior | Empirical witness |
|---|---|---|---|
| **Bias-scaling** | $\alpha \approx 1,\ \kappa=2$ | $\Delta(K) = \Theta(K)$, grows, clears multiplicity | Level 8: $+0.0078 \to +0.0108 \to +0.0134$, $p\ 0.017 \to 0.011 \to 0.0024$ |
| **Fading** | $\alpha \approx 1,\ \kappa=1$ | $\Delta(K) = \Theta(1)$, below detection floor | Gate-only: $+0.0044 \to +0.0026 \to +0.0030$, n.s. by $K=6$ |
| **Null** | $\alpha \approx 0,\ \kappa$ any | $\Delta \approx 0$ | scrambled control |
| **Anti-aligned** | $\alpha < 0,\ \kappa=2$ | $\Delta < 0$ (harvest on wrong bonds) | meas-only: $-0.027$, $0/12$ tasks |
| **Barren/overfit** | $\kappa \to K$ (global) | gradient $\sim 2^{-K}$, $n \sim 4^K$ | predicted; the regime TC-QIC avoids |
| **Classical-dominated** | any, with $n > n^\star$ | classical wins absolute (Thm 4.5) | $5$-$8$ pt lead at all $K$ |

The diagram is falsifiable: the **anti-aligned** region is a genuinely counterintuitive prediction (true-topology pooling doing *worse* than random), and the meas-only $-0.027$ result lands there, corroborating Lemma 4.2.

---

## Section 5 -- Boundary Conditions and the Classical Regime

### 5.1 Why classical wins (derived, not assumed)

Decompose the excess risk of each model as approximation + estimation error. Theorem 4.5 shows the quantum model carries two irreducible approximation terms -- the traced-out operator mass $\|g^\star_{\mathcal A^\perp}\|^2$ (operator geometry bottleneck) and the coarse-graining loss $\epsilon$ (topological bottleneck) -- while enjoying an $O(\sqrt{K/n})$ estimation term. The unconstrained classical model inverts this: near-zero approximation error, larger estimation error controlled by $n$. The **double bottleneck that gives TC-QIC its clean, non-absorbable bias is the same double bottleneck that caps its accuracy.** The 5-8-point gap is the arithmetic of this trade-off at Tox21's $(n, K, \epsilon)$.

### 5.2 The low-data prior did not materialize -- and the theory says why

One might expect (Theorem 4.4) a low-$n$ region where TC-QIC's variance advantage wins. Empirically this was **refuted** (the learning-curve study found the Level-8 bias is *data-hungry*, not sample-efficient). The theory accommodates this: the crossover $n^\star$ below which quantum wins exists only if the approximation-error deficit $\|g^\star_{\mathcal A^\perp}\|^2 + \epsilon$ is *smaller* than the classical model's low-$n$ estimation error. Here $\epsilon$ is large (aggressive coarse-graining) and the correlator signal $g^\star_{\mathcal S}$ is small (effect size $\le 1.1$ AUC pt), so $n^\star$ is pushed below the useful range -- the variance discount never overcomes the approximation deficit. **Boundary condition:** TC-QIC's regularization is real but too weak, at this coarse-graining, to buy a low-data regime. The remedy the theory suggests (Section 7) is milder coarse-graining (smaller $\epsilon$) and higher readout locality (larger $g^\star_{\mathcal S}$), not more data.

### 5.3 Kernel-alignment reading

Supervised QML models are kernel methods with a fixed quantum kernel (Schuld 2021); their generalization is set by kernel-target alignment (Kubler-Buchholz-Scholkopf 2021). TC-QIC's contribution is to *align* the induced kernel with molecular topology -- the structured > scrambled gap is a direct measurement of positive kernel-alignment increment. But an aligned weak kernel is still weaker than a flexible MLP on the same features: alignment improves the *ordering within the quantum family* without lifting the family above classical. This reconciles "bias is real" with "classical still wins" inside one formalism.

### 5.4 Failure modes, enumerated

The theory names four breakdowns, each with a trigger and a witness:

1. **Coarse-graining misspecification** ($\epsilon$ large): toxicophore split across clusters => topological prior wrong (Theorem 3.8). Diagnose via Phase A preservation rates.
2. **Placement-harvest mismatch** ($\alpha < 0$): entangler and readout use different graphs => negative bias (Lemma 4.2). Witness: meas-only $-0.027$.
3. **Readout over-localization** ($\kappa = 1$): marginals blind to the signal => fade (Lemma 3.10).
4. **Readout de-localization** ($\kappa \to K$): global observable => barren plateau and $n \sim 4^K$ (Master Theorem iii, contrapositive).

TC-QIC is the *interior* fixed point $\alpha = 1,\ \kappa = 2$ that avoids all four.

---

## Section 6 -- Connection to Empirical Results

Each theoretical object maps to a measured number.

| Theory | Prediction | Observed |
|---|---|---|
| **Lemma 1.1 / Def 1.3** (operator bottleneck, $\dim\mathcal A = \Theta(K)$ vs $4^K$) | trainable, no gradient collapse at small $K$ | learning ROC $0.53 \to 0.64$, no plateau at $K \le 6$ |
| **Theorem 2.7 / Cor 2.8** (absorbability = class identity) | zero population gap at absorbable levels | residual $= 0.00$ bit-exact, $K=4,6,8$ (L2/4) |
| **Cor 2.8(b)** (fixed-data non-absorbability) | Level 8 gap is genuine | stable, growing $\Delta$AUC that training never removes |
| **Theorem 2.2** (gap $= O(\sqrt{K/n})$, capacity-free) | struct $-$ scram independent of capacity | byte-identical params (Table I.4); gap persists |
| **Lemma 3.10** (single-qubit blindness) | gate-only fades, Level 8 grows | $+0.0044/0.0026/0.0030$ (n.s.) vs $+0.0078/0.0108/0.0134$ |
| **Prop 3.11** (linear K-law) | $\Delta_B(K) \approx 1.4 \times 10^{-3}K + 2.3 \times 10^{-3}$ | fit $R^2 \approx 0.996$; $p\ 0.017 \to 0.011 \to 0.0024$ |
| **Prop 3.11** (gate below power floor) | gate-only undetectable | $0.003 < 0.0066$ MDE at $80\%$ power |
| **Master Thm (i)** -- place | on-bond $|C|$ much greater than off-bond | $0.066$ vs $0.013$ = $5.1\times$; on-bond mass $0.71$ vs $0.34$ |
| **Master Thm (i)** -- harvest (Lemma 4.2) | true-$A$ pooling much greater than random-$A$ | $2.24\times$ vs $0.98\times$ uniform |
| **Master Thm (ii)** -- equivariance / topology-specific representation | structured features encode topology, tie on node feats | $\lambda_{\max}$ probe $R^2\ 0.072$ vs $0.040$ ($+80\%$); aromaticity ties |
| **Master Thm (iii)** -- trainability via locality (low-variance aggregates) | shot- and noise-robust difference | $+0.0184$ at 32 shots ($100\%$ positive); $+0.0182 \to +0.0168$ at $p = 0 \to 20\%$ |
| **Lemma 4.2** -- anti-alignment | true-$A$ pooling on wrong placement hurts | meas-only $-0.027$, $0/12$ tasks |
| **Theorem 4.5** -- classical dominance from bias-variance | classical leads by a fixed margin | $+5$-$8$ AUC pts at every $K$ |
| **Prop 3.11 saturation / multiplicity** | strongest cell survives correction | Level-8 $K=8$ Holm-adj $p = 0.017$ |

The shot- and noise-robustness deserve emphasis: they are the *empirical signature of the local-observable structure* in Master Theorem (iii). A structured $-$ scrambled **difference** over low-variance bond-pooled **aggregates** (sums of $\Theta(K)$ correlators) has its symmetric shot noise cancel in expectation, and multiplicative depolarization $(1-p)^2$ rescales both arms together -- exactly why $\Delta$AUC is flat in both shots and $p$. Locality is not a convenience; it is the mechanism of robustness.

---

## Section 7 -- Open Questions and Testable Predictions

The theory makes falsifiable predictions beyond the current data.

**P1 (Saturation of the K-law).** Proposition 3.11 predicts $\Delta_B(K)$ continues linearly ($\approx +0.0014$/qubit) through $K \approx 10$-$12$, then bends over near $K^\star \approx$ (heavy atoms)/(fragment size) as $\partial_K I(C(G);Y) \to 0$. *Test:* run $K = 10, 12, 16$; a strict linear continuation *refutes* the saturation mechanism, a plateau *confirms* it.

**P2 (Alignment knob traces the phase diagram).** Interpolate $A_\lambda = \lambda A_{\mathrm{true}} + (1-\lambda)A_{\mathrm{rand}}$. Lemma 4.2 predicts $\Delta(\lambda)$ monotone and approximately linear in $\lambda$ (overlap is linear in $A'$). *Test:* sweep $\lambda \in [0,1]$; nonlinearity or non-monotonicity falsifies the linear place-then-harvest identity.

**P3 (Locality knob $\kappa$).** Adding $3$-local bond-pooled correlators ($\kappa = 3$) should *increase* the bias iff genuine $3$-body toxicophores exist, at a trainability cost from higher-weight observables. *Test:* compare $\kappa \in \{1, 2, 3\}$; predicts a peak at the intrinsic body-order of the toxicophore.

**P4 (Param-matched equivariant classical -- the critical experiment).** Theorem 4.4 predicts a param-matched classical GNN with the *same* $O(K)$ sum-pooled message passing on the same $A$ should reproduce TC-QIC's generalization gap (same hypothesis class, Proposition 2.6) -- but only the quantum readout carries the specific two-qubit correlator alphabet. *Prediction:* param-matched $\mathrm{GNN}_{\mathrm{struct}} \approx$ TC-QIC$_{\mathrm{struct}}$, both $>$ scrambled, and both $<$ unconstrained MLP. A param-matched classical that *closes* the struct $-$ scram gap would refute the "non-classical message" clause (3) of Theorem 4.4.

**P5 (Differential feature-injection test).** Theorem 3.8 + operator bottleneck predict that adding within-cluster (high-graph-frequency) features helps the *classical* model (lowers $\epsilon$) but **not** the quantum readout (topology-bottlenecked to $\mathcal A(\mathcal O_8)$). *Test:* a crossed design; a quantum gain from high-frequency features would falsify the bottleneck's tightness.

**P6 (Trainability phase transition).** Master Theorem (iii) predicts that either deepening the circuit past $O(\log K)$ or globalizing the readout ($\kappa \to K$) re-introduces $2^{-K}$ gradient collapse. *Test:* measure $\operatorname{Var}[\partial_\theta\mathcal C]$ vs depth and vs $\kappa$; a sharp variance drop marks the boundary of the safe $O(K)$ regime.

**P7 (Universality of the scaling exponent).** If the growth is truly topological, the *slope-to-density ratio* $(\partial_K\Delta)/\bar d$ should be dataset-invariant across molecular families. *Test:* repeat on a second endpoint family (BBBP, ClinTox); an invariant normalized slope is strong evidence the mechanism is topology, not dataset idiosyncrasy -- and directly addresses the external-validity blocker.

**P8 (Low-$\epsilon$ regime recovers a quantum-favorable region).** Section 5.2 predicts that milder coarse-graining (larger $K$/atom, smaller $\epsilon$) plus higher $\kappa$ shrinks the approximation deficit and *lowers* the classical-crossover $n^\star$, possibly opening a genuine low-data quantum-favorable window on tasks where macro-topology *is* the whole signal (e.g. pure ring-system or scaffold classification). *Test:* construct a task with $g^\star_{\mathcal A^\perp} \approx 0$ by design and check whether TC-QIC matches or beats a param-matched classical there.

---

## Synthesis

TC-QIC unifies the phenomenological double bottleneck under one Lagrangian (Definition 1.4): the **topological bottleneck** is the ideal low-pass spectral projector $\Pi_{\le K}$ (Lemma 3.5), and the **operator geometry bottleneck** is the projection onto the $\Theta(K)$-dimensional $A$-weighted correlator subspace $\mathcal A(\mathcal O_8)$ (Definition 1.3). Their conjunction (i) tightens compression to a topology-aligned slice (Master Thm i), (ii) enforces $S_K$-equivariance with $O(\sqrt{K/n})$ generalization (Master Thm ii), and (iii) preserves poly-$K$ gradients via readout locality (Master Thm iii). The Absorbability Theorem (Theorem 2.7) certifies the control is non-vacuous exactly when structure enters as fixed data upstream of the trainable head -- the unique route that is also the scalable one. The same double bottleneck that produces a clean, non-absorbable, K-scaling inductive bias (structured > scrambled, growing to $p = 0.0024$ at $K = 8$) also caps absolute accuracy 5-8 points below an unconstrained classical model (Theorem 4.5) -- both facts now follow from a single bias-variance decomposition rather than being reported side by side. The theory is falsifiable on eight fronts (P1-P8), of which the param-matched equivariant classical (P4) and the second dataset (P7) are the load-bearing next experiments.

---

*Files consulted for exact statements and numbers: `docs/06_results_benchmarking.md` (Prop 1, Cor 2, Lemma 3; Tables I.1-III.2; power floor 0.0066), `docs/05_level8_report.md` (architecture, place-then-harvest, decomposition), and the extension results in `docs/09_extension_results.md` (shots, noise, mechanism 5.1x/2.24x, representation probe lambda_max 0.072/0.040).*

---

## Related Work and Theoretical Positioning

### Relation to Canatar et al. 2022 (Bandwidth Enables Generalization)

The table below maps every formal object in [Canatar2022] to its TC-QIC analog and records the relationship type and the precise statement bridging them.

| Canatar et al. object | TC-QIC analog | Relation | Precise statement |
|---|---|---|---|
| **Bandwidth c** (global scalar rescaling the encoder, $U(x) = R_x(cx)$) | Per-molecule adjacency $A = C(\mathrm{mol})$ entering both the graph-gated entangler $\mathrm{IsingXX}(A_{ij}\theta)$ and the bond-pooled readout $B_A$ | GENERALIZES + REPLACES mechanism | $c$ is one scalar fixed once for the whole dataset; $A$ is an input-conditioned, non-trainable structured operator. TC-QIC is a "per-molecule adaptive bandwidth." A scalar/trainable bandwidth knob placed in front of a variational head is absorbable (Thm 2.7); TC-QIC works only because $A$ enters as fixed per-molecule data upstream of the trainable layer (Cor 2.8(b)). |
| **Kernel $k(x,x') = \mathrm{Tr}(\rho(x)\rho(x'))$** (full-state fidelity kernel, rank up to $4^K$) | Restricted feature map $\phi_\mathcal{O}(\rho) = (\mathrm{Tr}[O_a \rho])$ and its induced kernel $k_\mathcal{O}(x,x') = \langle\phi_\mathcal{O}(x), \phi_\mathcal{O}(x')\rangle$, of rank $\Theta(K)$ | REPLACES (COMPLEMENTS via Sec 5.3) | TC-QIC deliberately discards the full fidelity kernel for a low-rank measurement kernel supported on the $\Theta(K)$-dimensional accessible operator subspace $\mathcal{A}(\mathcal{O}_8)$. Their kernel is fixed; ours sits downstream of trainable $\theta$. The kernel reading survives only as the alignment lens of Sec 5.3. |
| **Task-model alignment $C(l)$** (cumulative target power in first $l$ kernel modes) | Signal-readout alignment $\eta_\mathcal{O}(K) = \dim(\mathcal{A}(\mathcal{O}) \cap \mathcal{S}(K))/\dim \mathcal{S}(K)$ (Eq 3.3); topology alignment $\alpha$ (Eq 4.3); epsilon-sufficiency (Def 3.7) | EXTENDS | We adopt "generalization is governed by target/model-mode alignment" and recast it geometrically: overlap of the label-relevant operator subspace $\mathcal{S}(K)$ with the accessible subspace $\mathcal{A}(\mathcal{O})$, plus a second, architecture-specific place-vs-harvest overlap $\alpha$ that Canatar has no analog for. |
| **Kernel spectrum $\eta_k$** (soft polynomial decay with bandwidth; flat without) | $\dim \mathcal{A}(\mathcal{O}_8) = \Theta(K)$ out of $4^K$ (Def 1.3): a hard rank truncation of operator space; separately, the graph-Laplacian spectrum truncated by $\Pi_{\le K}$ (Lemma 3.5) | EXTENDS (analogous role, different mechanism) | Canatar shapes one spectrum by rescaling its decay; TC-QIC controls the support/rank of the operator spectrum by projection, and additionally acts on a second spectrum (the graph Laplacian) via low-pass coarse-graining. Same "keep the low/aligned modes" principle, realized as projection rather than bandwidth. |
| **No-generalization result** ($c=1$ => flat spectrum => $O(3^n)$ samples/mode) | Corollary 2.3: full-Hilbert readout has feature norm $B = \Theta(2^K)$, Rademacher $O(2^K W/\sqrt{n})$, demanding $n = \Omega(4^K)$ | EXTENDS (same conclusion, new setting/proof object) | The shared load-bearing result: an unrestricted embedding cannot generalize. Their mechanism is kernel-spectrum flatness (exponential concentration); ours is operator-space dimension via Rademacher. The full $4^K$ readout is the operator-space image of their $c=1$ regime. |
| **Polynomial learnability** ($P \sim n^l$ samples for mode $l$ once bandwidth is on) | Theorem 2.2: $R(h) \le \hat{R} + O(W\sqrt{K/n})$; $n = O(K)$ suffices | EXTENDS | Both give polynomial sample complexity once the bias is imposed. Their route is spectral decay + mode counting (average-case); ours is a bounded low-dimensional feature map (worst-case uniform convergence). Same conclusion class, different tool. |
| **Replica-method generalization formula** $E_g = \kappa^2/(1-\gamma) \cdot \sum_k a_k^2/(\kappa + \alpha \eta_k)^2$ | none | NOT PRESENT | TC-QIC currently has only the Rademacher upper bound; it has no average-case, spectrum-resolved $E_g$. This is the prime candidate for incorporation (see Section C below). |
| **Inductive bias = controlling the kernel integral-operator spectrum via c** | Q-IB Lagrangian (Def 1.4): bias = restriction of the channel feasibility set, realized as $\mathcal{A}(\mathcal{O})$ (Def 1.3) + graph low-pass $\Pi_{\le K}$ (Lemma 3.5) | GENERALIZES | Canatar's bias is a point on a single scalar spectrum-control axis. TC-QIC gives an information-theoretic definition of bias (feasible-set restriction under a fixed measurement CPTP channel) and instantiates it as a double bottleneck (operator geometry + topology), of which scalar bandwidth is a degenerate special case. |

#### What TC-QIC Builds On from Canatar et al.

1. **Alignment governs generalization (their $C(l)$).** We adopt their central thesis -- that generalization is set by the alignment between the target and the model's accessible modes, not by raw expressivity -- and recast it as operator-subspace alignment. Their cumulative power $C(l)$ becomes our $\eta_\mathcal{O}(K)$ (share of the placed-signal subspace $\mathcal{S}(K)$ that lies in the accessible subspace $\mathcal{A}(\mathcal{O})$) and our place-vs-harvest overlap $\alpha$. Proposition 3.11's K-scaling law is, at bottom, "aligned signal mass minus a complexity penalty," which is their alignment-vs-spectrum trade-off transplanted to a variational readout.

2. **The necessity of inductive bias (their no-generalization result).** We build directly on their diagnosis that a free/unbiased embedding cannot generalize (flat spectrum, exponential sample cost). Corollary 2.3 is the deliberate operator-space restatement: the all-$4^K$-Paulis readout is our $c=1$ analog, with the same exponential-sample-complexity verdict. Their result is our motivation for imposing a hard operator bottleneck rather than an argument we re-prove from scratch.

3. **The "engineer the spectrum, not the expressivity" paradigm.** We inherit their framing that the correct design object is the accessible-mode structure of the model. TC-QIC simply moves the control knob from a scalar bandwidth on the encoder to the geometry of the measurement channel (and the graph coarse-graining resolution $K$).

4. **The kernel-alignment reading (Sec 5.3).** Our claim that the structured-vs-scrambled gap is a positive kernel-alignment increment is stated in exactly their (and Kuebler et al.'s) vocabulary of kernel-target alignment; the induced correlator kernel $k_\mathcal{O}$ is the bridge object.

#### What TC-QIC Extends Beyond Canatar et al.

1. **Variational feature map, not a fixed kernel.** Canatar analyze fixed-kernel regression (no trainable parameters). TC-QIC places trainable $\theta$ upstream of the readout. This is not cosmetic: it is what makes the entire absorbability question exist.

2. **Absorbability Theorem (2.7) -- a gap their framework cannot see.** Because their bandwidth is a hyperparameter of a fixed kernel, absorbability never arises for them. But any attempt to port "bandwidth as bias" into a trainable circuit is provably vacuous: a free trainable linear map upstream re-absorbs any fixed input reparametrization at zero cost ($\mathcal{H}_{\mathrm{struct}} = \mathcal{H}_{\mathrm{scram}}$, bit-exact). TC-QIC contributes the exact non-absorbability criterion (Cor 2.8): the structured quantity must enter as fixed per-molecule data multiplying a physical observable, upstream of the only trainable layer. This is a genuinely new no-go/when-valid result with no Canatar analog.

3. **Per-molecule adaptive bandwidth.** Their $c$ is one global scalar; our $A = C(\mathrm{mol})$ is input-conditioned and topology-derived. The bias is not a single operating point but a molecule-dependent projection -- a strictly richer object.

4. **Chemical-topology prior with proven task symmetry.** Canatar are task-agnostic. TC-QIC injects the molecular graph, coarse-grains it (spectral low-pass, Lemma 3.5), and proves the induced $S_K$-equivariance is aligned with the true label symmetry (Prop 3.3, Axiom 3.2). None of this exists in the kernel-bandwidth picture.

5. **CPTP / Q-IB framing.** TC-QIC defines inductive bias information-theoretically: the measurement is an entanglement-breaking CPTP channel whose data-processing/Holevo ceiling sets the extractable information (Lemma 1.1), and the bias is the restriction of the channel feasibility set (Def 1.4). Canatar have no bottleneck, channel, or Holevo-ceiling framing.

6. **Operator-geometry mechanism (hard rank vs soft decay).** Their bandwidth reshapes the decay of an otherwise-full spectrum; TC-QIC imposes a hard low-rank projection ($\dim \mathcal{A} = \Theta(K)$ out of $4^K$) plus a second Laplacian low-pass. Two spectra, two projections, versus their one spectrum, one scalar.

7. **Trainability and equivariance results.** Barren-plateau resistance via readout locality (Master Thm iii, Cerezo-type local-cost argument) and the equivariance regularizer (Thm 2.4) are outside the kernel setting entirely -- a fixed kernel has no gradients to vanish and no weights to tie.

8. **Quantum message-passing correspondence (Prop 2.6).** Identifying the bond-pooled two-qubit connected correlator as one MPNN aggregation step with a quantum message alphabet is a bridge Canatar do not build.

#### What Canatar et al. Provide That TC-QIC Should Incorporate

1. **Their replica-method $E_g$ formula (average-case, sharper than our Rademacher bound).** Theorem 2.2 gives only a worst-case $O(W\sqrt{K/n})$ upper bound. Specializing their $E_g = \kappa^2/(1-\gamma) \cdot \sum_k a_k^2/(\kappa + \alpha \eta_k)^2$ to the $\Theta(K)$-rank correlator kernel would yield an average-case learning curve -- a direct, testable prediction of Delta AUC versus $n$. This is exactly what the project needs to explain the empirically observed data-hungry behavior (the low-data prior that did not materialize, Sec 5.2): the replica curve would predict the crossover $n^\star$ quantitatively instead of asserting it.

2. **Explicit spectrum-decay characterization inside the accessible subspace.** TC-QIC counts $\dim \mathcal{A} = \Theta(K)$ but says nothing about the eigenvalue decay of the induced correlator kernel within that subspace. Importing their $\eta_k$ analysis would upgrade Prop 3.11 from a dimension count to a spectral-decay law and would give a spectral origin for the AUC ceiling (0.61-0.66) as spectrum saturation rather than only as coarse-graining loss $\epsilon$.

3. **Bandwidth-as-spectrum-control framing for the K-scaling law.** Recasting $A = C(\mathrm{mol})$ as setting an effective per-molecule bandwidth $c_{\mathrm{eff}}(\mathrm{mol})$, and $K$ (coarse-graining resolution) as a global bandwidth knob, would let us predict how the induced spectrum -- hence learnability -- scales with $K$, tying the linear law $\Delta_B(K) \approx 1.4 \times 10^{-3} K + 2.3 \times 10^{-3}$ to a spectral mechanism and sharpening the saturation prediction at $K^\star$.

4. **Their cumulative power $C(l)$ as a measurable diagnostic.** Computing $C(l)$ on the induced correlator kernel would turn the qualitative Sec 5.3 alignment claim into a quantitative, pre-registrable predictor: high $C(l)$ at small $l$ should predict where the topology prior helps.

---

### Quantum Generalization and Kernel Learning

The generalization behavior of quantum models is now understood to hinge on the spectrum of the induced kernel and its alignment with the target, rather than on circuit expressivity. [Canatar2022] make this sharp for quantum kernel machines: without a bandwidth hyperparameter the fidelity kernel's spectrum is exponentially flat ($\eta_{\max} \sim 2^{-n}$), forcing $O(3^n)$ samples per mode and precluding generalization, whereas a bandwidth $c \sim n^{-1/2}$ induces polynomial spectral decay and polynomial learnability. Our work extends [Canatar2022] from fixed-kernel regression to a trainable variational readout, and replaces their single global scalar bandwidth with a per-molecule, non-trainable adjacency $A = C(\mathrm{mol})$ that acts as an adaptive bandwidth. Where they shape the decay of one full-rank spectrum, we impose a hard low-rank projection onto a $\Theta(K)$-dimensional accessible operator subspace (our Corollary 2.3 is the operator-space restatement of their $c=1$ no-generalization result), and we act on a second, graph-Laplacian spectrum via low-pass coarse-graining.

This alignment-centric view is corroborated by [Huang2021], who show classical data can erase a putative quantum advantage and that task-model alignment -- formalized through their projected quantum kernel over reduced observables -- is decisive; and by [Kuebler2021], who argue quantum measurements supply an inductive bias for molecular chemistry. Our work differs from [Huang2021] by conditioning the accessible observables on molecular topology per input rather than defining a single projected kernel, and from [Kuebler2021] by reading two-qubit connected correlators pooled along real bonds (a $\Theta(K)$-dimensional graph-weighted subspace) rather than generic Born-rule measurements on small molecules, and by proving that this specific readout makes the structured-vs-scrambled control non-absorbable. The exponential-concentration failure mode we avoid is the same one flagged for generic quantum kernels [Thanasilp2022], which our operator bottleneck sidesteps by construction.

On finite-sample guarantees, [Caro2022] give Rademacher/covering-number bounds for variational QML that scale with the number of gates or trainable parameters. Our Theorem 2.2 differs by deriving the bound from the readout dimension $\Theta(K)$ rather than the gate count, yielding $O(W\sqrt{K/n})$ and an explicit $\Omega(4^K) \to O(K)$ sample-complexity saving relative to an unrestricted readout. We regard the replica-style average-case machinery of [Bordelon2020] as complementary and sharper than our worst-case bound, and we identify it as the natural tool for turning our K-scaling law into a predicted learning curve.

---

### Inductive Bias in Quantum Circuits

[McClean2018] established that random deep circuits with global cost observables suffer barren plateaus -- gradient variance vanishing as $2^{-K}$ -- and [Cerezo2021] showed that local cost functions in shallow circuits retain at-most-polynomially-vanishing gradients. Our Master Theorem (iii) invokes exactly this local-cost mechanism: the bond-pooled readout is a sum of $\Theta(K)$ at-most-2-local observables at $O(1)$ depth, so gradients stay $\mathrm{poly}(1/K)$. We differ from this line by using locality not merely to preserve trainability but as the source of the inductive bias itself (the operator-geometry bottleneck) and of shot/noise robustness, and by clarifying that barren-plateau resistance comes from the measurement, not from any restriction on the unitary's Hilbert-space exploration.

[Abbas2021] quantify a quantum model's capacity via effective dimension and relate it to trainability and expressibility. Our work differs by making the accessible dimension explicit and architectural -- $\dim \mathcal{A}(\mathcal{O}_8) = \Theta(K)$ out of $4^K$ -- so that the capacity is fixed by the readout geometry rather than estimated post hoc from the Fisher information, and by proving that this $\Theta(K)$ support is precisely the graph-weighted correlator subspace aligned to the task symmetry. The broader equivariant-QML program ([Larocca2022]; [Meyer2023]; [Nguyen2024]) obtains symmetry by constraining gates; we obtain $S_K$-equivariance from measurement geometry alone (Theorem 2.4), and we complement [Schuld2021enc], who show the encoding fixes the accessible function family, with the dual statement that the readout fixes the accessible operator subspace.

[Schuld2022] argue that quantum advantage is the wrong goal and that characterizing quantum inductive bias is the productive question. We adopt this reframing wholesale: our claim is existence-and-scaling of a topology bias, explicitly not quantum-over-classical (a capacity-unconstrained classical MLP still leads by 5-8 AUC points, which our Theorem 4.5 derives from the same bias-variance trade-off). Our contribution beyond their position paper is a concrete, falsifiable bias with a growth law and a provably non-vacuous control (Theorem 2.7) -- the absorbability result, which to our knowledge has no prior analog and which shows that the field's standard structured-vs-scrambled test is mathematically empty unless the structure enters as fixed data upstream of the trainable head.

---

### Graph Neural Networks and Molecular ML

Message-passing neural networks [Gilmer2017] set the template for molecular property prediction: iterated neighborhood aggregation $h_i' = \varphi(h_i, \sum_{j \in N(i)} \psi(h_i, h_j, A_{ij}))$. Our Proposition 2.6 shows the Level-8 readout $b[i] = \sum_j A_{ij} C_{ij}$ is exactly one such aggregation step; we differ from [Gilmer2017] only in the message alphabet, which is a genuine two-qubit connected correlator (an entanglement-carrying, non-classical quantity) rather than a learned function of node features. This makes TC-QIC a quantum instance of message passing, and its inductive bias formally identical to a GNN's at the level of the aggregation operator. Related quantum-graph constructions ([Verdon2019]; [Mernyei2022]; [Skolik2023]) build equivariant circuits for graphs; we differ by placing the equivariance in the measurement and by pooling along chemically coarse-grained bonds.

[Xu2019] characterize GNN expressivity via the Weisfeiler-Leman test and show sum-pooling is maximally discriminative among neighborhood aggregators. Our sum-pooled, $S_K$-equivariant readout inherits this WL-aligned aggregation, and Proposition 3.3 supplies the first-principles reason it is appropriate: toxicity is $S_n$-invariant, so the orbit is a sufficient statistic and any non-equivariant model wastes capacity. We differ from the expressivity literature by tying the coarse-graining to spectral graph theory: the molecular graph is projected onto its bottom-$K$ Laplacian eigenspace (Lemma 3.5), so the topological bottleneck is an ideal low-pass filter in the sense of spectral clustering ([vonLuxburg2007]; [ShiMalik2000]) and spectral GNNs ([Bruna2014]; [Defferrard2016]). This gives a precise definition of the "high-frequency atomic noise" removed before encoding and an explicit sufficiency criterion (Theorem 3.8) for when the coarse-graining preserves a toxicophore.

---

### Information Bottleneck

The information bottleneck [Tishby2000] frames representation learning as the compression-relevance trade-off $\min I(T;X) - \beta I(T;Y)$. We build our theory on this Lagrangian but differ in one decisive respect. In the classical IB the channel $p(t \mid x)$ is free -- the optimizer ranges over all stochastic maps -- so the IB itself imposes no architectural bias. In our Quantum IB (Definition 1.4), the optimization domain is restricted to circuit parameters $\theta$ at a fixed measurement structure $\mathcal{O}$: the readout is an entanglement-breaking CPTP channel whose data-processing and Holevo ceilings cap $I(X;T)$ at $\Theta(K)$ coordinates (Lemma 1.1). The inductive bias is therefore precisely the restriction of the channel feasibility set -- the choice of $\mathcal{O}$ -- which has no counterpart in the free-channel classical IB.

Our work thus differs from the IB literature ([Tishby2015]; [Achille2018]) by turning the bottleneck from an objective into a hard structural constraint realized by physical measurement, and by splitting it into two interacting bottlenecks: a topological one (spectral low-pass on the graph, Lemma 3.5) and an operator-geometry one (projection onto the $\Theta(K)$-dimensional accessible subspace, Definition 1.3). Where classical IB studies the trade-off along $\beta$, we study it along two order parameters -- topology alignment $\alpha$ and readout locality $\kappa$ -- and show (Master Theorem, Theorem 4.5) that the same double bottleneck that yields a clean, non-absorbable, K-scaling bias also caps absolute accuracy below an unconstrained classical model. This positions TC-QIC as a physically instantiated, symmetry-aware IB, distinct both from the classical objective and from the fixed-kernel spectral picture of [Canatar2022].

---

### Per-Theorem Citation Map

The table below records, for each TC-QIC theorem, the closest prior work and the precise novel contribution.

| Theorem / Result | Closest prior work | What is new |
|---|---|---|
| **Lemma 1.1** (measurement is a bottleneck) | [Tishby2000] (classical IB); Holevo bound [Holevo1973]; quantum DPI [NielsenChuang2000]; [Kuebler2021] | Casting the observable choice itself as the IB constraint -- the measurement CPTP channel sets the extractable-information ceiling and is the inductive bias, rather than being a readout detail |
| **Def 1.3** (accessible operator subspace) | [Schuld2021] (QML models are kernel methods); [Huang2020] (classical shadows); [Huang2021] (projected quantum kernel) | Explicit $\dim \mathcal{A}(\mathcal{O}_8) = \Theta(K)$ out of $4^K$ count and identification of that subspace's span as the precise locus of inductive bias |
| **Prop 1.2** (trace-out as structured prior) | [Mitchell1980] (bias = hypothesis class restriction); Fisher-Neyman sufficiency; [CohenWelling2016] (equivariance-as-bias) | The traced-out operator complement $\mathcal{A}^\perp$ is a deterministic architectural prior, not stochastic noise, with the lossless-iff-$g^\star_{\mathcal{A}^\perp}=0$ dichotomy (zero-cost vs variance-reducing prior) |
| **Theorem 2.2** (Rademacher bound) | [Caro2022] (QML generalization); [BartlettMendelson2002] (Rademacher for linear classes); [Abbas2021] (effective dimension) | Bound driven by readout dimension $\Theta(K)$, not gate/parameter count -- yields $O(W\sqrt{K/n})$ polynomial in qubits by construction; Caro et al. scale with gate count, ours with readout locality, a tighter route for this architecture |
| **Theorem 2.4** ($S_K$ equivariance) | [Larocca2022] (group-invariant QML); [Meyer2023]; [Nguyen2024]; [CohenWelling2016]; [Bronstein2021] | Equivariance obtained from measurement geometry (bond-pooled readout) rather than from constraining gates or weight-tying -- it emerges from the observable choice |
| **Prop 2.6** (quantum message passing) | [Gilmer2017] (MPNN); [Verdon2019]; [Mernyei2022]; [Skolik2023] | Proving the bond-pooled two-qubit connected correlator is exactly one MPNN aggregation step, differing only in the message alphabet (an entanglement-carrying correlator, not a learned node-feature function) |
| **Theorem 2.7** (Absorbability) | [Schuld2021enc] (encoding determines accessible function family); reparametrization/gauge redundancy [Dinh2017]; [PerezSalinas2020] (data re-uploading) | No close prior: a formal function-class-identity theorem showing the standard structured-vs-scrambled control is vacuous under an upstream trainable linear map, together with the checkable non-absorbability criterion (a)/(b) |
| **Lemma 3.5** (spectral low-pass) | [vonLuxburg2007] (spectral clustering tutorial); [ShiMalik2000] (normalized cuts); [Shuman2013] (graph signal processing); [Bruna2014]; [Defferrard2016] | Casting molecular coarse-graining as the ideal low-pass projector $\Pi_{\le K}$ onto the low graph-Laplacian modes, giving "high-frequency atomic noise" a precise definition as the discarded high-frequency band feeding the quantum circuit |
| **Theorem 3.8** (conditional sufficiency) | [Tishby2000]; [Achille2018] (minimal sufficiency); Fisher-Neyman sufficiency; toxicophore/structural-alert chemistry | Epsilon-sufficiency criterion tied to whether coarse-graining splits a toxicophore across a cluster boundary, giving an explicit boundary condition and deriving the observed AUC ceiling |
| **Prop 3.11** (K-scaling law) | [Canatar2022] (spectrum-controlled scaling); [Bordelon2020] (spectral learning curves); [Bahri2021] (neural scaling laws) | A bias scaling law $\Delta(K) \sim \Theta(K)$ -- the inductive-bias effect grows with qubit count ($K = 4, 6, 8$: Delta AUC $0.0078 \to 0.0108 \to 0.0134$; $p\ 0.017 \to 0.011 \to 0.0024$; $R^2 \approx 0.996$) -- a growth-with-resource law derived as aligned mass minus complexity penalty, rather than an error-decay law |
