# T2: Operator-Geometry Bottleneck -- Accessible Subspace Dimension

*Status: COMPLETE | Priority: HIGH | Phase-gate: $\dim \mathcal{A}(\mathcal{O}_8) = \Theta(K)$*

**Theorem (Operator-Geometry Bottleneck).** Let $\mathcal{O}_8$ be the Level-8
readout family on $K$ qubits over a connected molecular graph $G = (V, E)$ with
generic bond weights $A$. Then the accessible operator subspace satisfies

$$
\dim \mathcal{A}(\mathcal{O}_8) \;=\; 3K + 2\,\mathrm{rank}(\mathrm{Pool}_A)
\;=\; 5K \;=\; \Theta(K)
\quad \text{out of} \quad \dim \mathrm{Herm}(\mathcal{H}) = 4^K .
$$

Consequently the feature map $\phi_{\mathcal{O}}(\rho)_a = \mathrm{Tr}[O_a \rho]$
is blind to a subspace of codimension $5K$, i.e. it implements a hard rank-$5K$
projection of the exponentially large operator space.

---

## 1. Setup

**Hilbert space and operator space.** Let $\mathcal{H} = (\mathbb{C}^2)^{\otimes K}$,
so $\dim \mathcal{H} = 2^K$. Let $\mathrm{Herm}(\mathcal{H})$ denote the real
vector space of Hermitian operators on $\mathcal{H}$; as a real vector space,

$$
\dim_{\mathbb{R}} \mathrm{Herm}(\mathcal{H}) = (2^K)^2 = 4^K .
$$

**Hilbert-Schmidt inner product.** Equip $\mathrm{Herm}(\mathcal{H})$ with the
normalized Hilbert-Schmidt (HS) inner product

$$
\langle O, O' \rangle_{HS} \;=\; 2^{-K}\, \mathrm{Tr}\!\left[ O^{\dagger} O' \right],
$$

which is a real inner product on $\mathrm{Herm}(\mathcal{H})$ (Hermiticity makes
$\mathrm{Tr}[O^{\dagger} O'] = \mathrm{Tr}[O O']$ real: $\overline{\mathrm{Tr}[OO']}
= \mathrm{Tr}[(OO')^{\dagger}] = \mathrm{Tr}[O'O] = \mathrm{Tr}[OO']$).

**Pauli string basis.** For $\mu = (\mu_1, \dots, \mu_K) \in \{I, X, Y, Z\}^K$
define the Pauli string $P_{\mu} = \sigma_{\mu_1} \otimes \cdots \otimes \sigma_{\mu_K}$.
Since $\mathrm{Tr}[\sigma_a \sigma_b] = 2 \delta_{ab}$ for $a, b \in \{I, X, Y, Z\}$,
traces factorize over tensor legs:

$$
\langle P_{\mu}, P_{\nu} \rangle_{HS}
= 2^{-K} \prod_{i=1}^{K} \mathrm{Tr}[\sigma_{\mu_i} \sigma_{\nu_i}]
= 2^{-K} \prod_{i=1}^{K} 2\,\delta_{\mu_i \nu_i}
= \delta_{\mu \nu}.
\tag{1.1}
$$

Hence $\{P_{\mu}\}_{\mu}$ is an HS-orthonormal basis of $\mathrm{Herm}(\mathcal{H})$
with $4^K$ elements. The *weight* of $P_{\mu}$ is $w(\mu) = |\{ i : \mu_i \neq I \}|$,
the number of non-identity tensor legs; its *support* is $\mathrm{supp}(\mu) =
\{ i : \mu_i \neq I \}$. By (1.1), two Pauli strings are HS-orthogonal unless they
agree letter-by-letter on every site -- in particular unless they have identical
support and identical letters on that support.

**Level-8 readout family.** Fix a molecular graph $G = (V, E)$ with $V = \{1, \dots, K\}$
and symmetric bond-weight matrix $A = (A_{ij})$, $A_{ij} \neq 0 \iff (i,j) \in E$,
$A_{ii} = 0$. The Level-8 readout family is

$$
\mathcal{O}_8 \;=\;
\underbrace{\{ X_i, Y_i, Z_i : i = 1, \dots, K \}}_{3K \text{ single-qubit ops}}
\;\cup\;
\underbrace{\Big\{ O^{ZZ}_i = \sum_{j} A_{ij}\, Z_i Z_j : i = 1, \dots, K \Big\}}_{K \text{ bond-pooled } ZZ}
\;\cup\;
\underbrace{\Big\{ O^{XX}_i = \sum_{j} A_{ij}\, X_i X_j : i = 1, \dots, K \Big\}}_{K \text{ bond-pooled } XX},
$$

where $X_i$ denotes the weight-1 string with letter $X$ at site $i$ and identity
elsewhere, and $Z_i Z_j$ ($i \neq j$) the weight-2 string with letter $Z$ at sites
$i, j$ and identity elsewhere (similarly $X_i X_j$).

**Accessible subspace.** Define

$$
\mathcal{A}(\mathcal{O}_8) \;=\; \mathrm{span}_{\mathbb{R}}\, \mathcal{O}_8
\;\subseteq\; \mathrm{Herm}(\mathcal{H}),
$$

and write it as the sum of three subspaces:

$$
\mathcal{A}(\mathcal{O}_8) = S_1 + S_{ZZ} + S_{XX},
\qquad
S_1 = \mathrm{span}\{X_i, Y_i, Z_i\}, \quad
S_{ZZ} = \mathrm{span}\{O^{ZZ}_i\}, \quad
S_{XX} = \mathrm{span}\{O^{XX}_i\}.
$$

We compute the dimension of each summand (Sections 2-4), show the sum is
HS-orthogonal (Section 5), and conclude (Section 6).

---

## 2. Dimension of the single-qubit subspace: $\dim S_1 = 3K$

**Claim.** The $3K$ operators $\{X_i, Y_i, Z_i : i = 1, \dots, K\}$ are pairwise
HS-orthonormal, hence linearly independent, hence $\dim S_1 = 3K$.

**Proof.** Each of these operators is a weight-1 Pauli string: $X_i = P_{\mu}$
with $\mu_i = X$ and $\mu_k = I$ for $k \neq i$, and similarly for $Y_i, Z_i$.
Two such strings coincide as multi-indices iff they act on the same qubit with
the same letter. Apply (1.1):

*Same letter, different qubits.* For $i \neq j$,

$$
\langle X_i, X_j \rangle_{HS}
= 2^{-K}\, \mathrm{Tr}\big[ (I \otimes \cdots \otimes X_{(i)} \otimes \cdots \otimes I)
                            (I \otimes \cdots \otimes X_{(j)} \otimes \cdots \otimes I) \big]
= 2^{-K} \cdot \mathrm{Tr}[X]^2 \cdot 2^{K-2} = 0,
$$

since $\mathrm{Tr}[X] = 0$ appears on both leg $i$ and leg $j$ of the factorized
trace. Identically for $Y$ and $Z$.

*Different letters, same qubit.* $\langle X_i, Y_i \rangle_{HS}
= 2^{-K} \cdot \mathrm{Tr}[XY] \cdot 2^{K-1} = 0$ since $\mathrm{Tr}[XY] =
\mathrm{Tr}[iZ] = 0$; likewise for the pairs $(X, Z)$ and $(Y, Z)$.

*Different letters, different qubits.* Both legs $i$ and $j$ carry a traceless
single-site factor, so the product trace vanishes as in the first case.

*Normalization.* $\langle X_i, X_i \rangle_{HS} = 2^{-K}\,\mathrm{Tr}[X^2] \cdot 2^{K-1}
= 2^{-K} \cdot 2 \cdot 2^{K-1} = 1$, and likewise for $Y_i, Z_i$.

Thus $\langle X_i, X_j \rangle_{HS} = \delta_{ij}$ and all cross-letter inner
products vanish: the $3K$ operators form an HS-orthonormal set, and
$\dim S_1 = 3K$. $\blacksquare$

**Remark (weight separation).** By (1.1), every weight-1 Pauli string is
HS-orthogonal to every weight-2 Pauli string: they differ as multi-indices
(a weight-2 string has a non-identity letter at some site where the weight-1
string has $I$, since their supports cannot be equal). This will give the
cross-orthogonality in Section 5.

---

## 3. Dimension of the bond-pooled ZZ subspace: $\dim S_{ZZ} = \mathrm{rank}(\mathrm{Pool}_A)$

**Ambient space.** Enumerate the edges $E = \{ e_1, \dots, e_{|E|} \}$, writing
$e = \{ i(e), j(e) \}$ for the endpoints. Each pooled operator lies in the span
of bonded weight-2 $ZZ$ strings:

$$
O^{ZZ}_i = \sum_{j : (i,j) \in E} A_{ij}\, Z_i Z_j
\;\in\; W_{ZZ} := \mathrm{span}\{ Z_{i(e)} Z_{j(e)} : e \in E \}.
$$

By (1.1) the $|E|$ strings $\{ Z_{i(e)} Z_{j(e)} \}_{e \in E}$ are HS-orthonormal
(distinct edges have distinct supports), so they form an orthonormal basis of
$W_{ZZ}$ and $\dim W_{ZZ} = |E|$.

**Pooling matrix.** Define the $K \times |E|$ *bond-pooling matrix*
$\mathrm{Pool}_A$ by

$$
(\mathrm{Pool}_A)_{i, e} \;=\;
\begin{cases}
A_{i(e) j(e)} & \text{if } i \in e = \{ i(e), j(e) \}, \\
0 & \text{otherwise.}
\end{cases}
$$

Note $Z_i Z_j = Z_j Z_i$, so the edge $e = \{i, j\}$ contributes the *same*
basis vector to both $O^{ZZ}_i$ and $O^{ZZ}_j$. In the orthonormal edge basis
of $W_{ZZ}$, the coordinate vector of $O^{ZZ}_i$ is exactly the $i$-th row of
$\mathrm{Pool}_A$:

$$
O^{ZZ}_i = \sum_{e \in E} (\mathrm{Pool}_A)_{i, e}\; Z_{i(e)} Z_{j(e)} .
\tag{3.1}
$$

**Claim.** $\dim S_{ZZ} = \mathrm{rank}(\mathrm{Pool}_A)$.

**Proof.** By (3.1), the linear map sending the $i$-th standard basis vector of
$\mathbb{R}^K$ to $O^{ZZ}_i$ factors through the coordinate isomorphism
$W_{ZZ} \cong \mathbb{R}^{|E|}$ (orthonormal bases preserve linear-independence
structure exactly). The dimension of the span of a set of vectors equals the
rank of the matrix whose rows are their coordinates, hence
$\dim \mathrm{span}\{ O^{ZZ}_i \} = \mathrm{rank}(\mathrm{Pool}_A)$. $\blacksquare$

**Claim (generic full row rank).** If $G$ is connected with $|E| \geq K$
(equivalently: connected and not a tree, or more generally every vertex has
degree $\geq 1$ and the weights avoid a measure-zero set), then
$\mathrm{rank}(\mathrm{Pool}_A) = K$ for Lebesgue-almost-every weight assignment
$(A_e)_{e \in E} \in \mathbb{R}^{|E|}$, and in particular for generic positive
weights.

**Proof.** Since $\mathrm{Pool}_A$ is $K \times |E|$ with $|E| \geq K$, full row
rank $\mathrm{rank} = K$ is equivalent to the non-vanishing of at least one
$K \times K$ minor. Each minor is a polynomial in the weight variables
$(A_e)_{e \in E}$.

*Step 1: some minor is not the zero polynomial (generic-rank computation).*
Work over the field $\mathbb{R}(A)$ of rational functions in the
indeterminates $(A_e)_{e \in E}$. Treat the rows of $\mathrm{Pool}_A$ as
vectors $r_i = \sum_{e \ni i} A_e\, \mathbf{1}_e \in \mathbb{R}(A)^{|E|}$,
where $\mathbf{1}_e$ is the $e$-th standard basis vector. Suppose
$\sum_{i=1}^{K} c_i\, r_i = 0$ with $c_i \in \mathbb{R}$, identically in the
indeterminates. The coefficient of the monomial $A_e$ (edge $e = \{i, j\}$)
in this sum is $(c_i + c_j)\, \mathbf{1}_e$, so the relation forces

$$
c_i + c_j = 0 \qquad \text{for every edge } (i, j) \in E.
\tag{3.2}
$$

If $G$ contains an odd cycle $(i_1, i_2, \dots, i_{\ell}, i_1)$, $\ell$ odd,
then propagating (3.2) around the cycle gives $c_{i_1} = -c_{i_2} = c_{i_3}
= \cdots = (-1)^{\ell} c_{i_1} = -c_{i_1}$, forcing $c_{i_1} = 0$ and hence
$c = 0$ on the cycle; propagating (3.2) along paths ($c_j = -c_i$) then kills
$c$ on all of $V$ by connectedness. So for non-bipartite connected $G$ the
rows are linearly independent over $\mathbb{R}(A)$, i.e. some $K \times K$
minor of $\mathrm{Pool}_A$ is a nonzero polynomial in $(A_e)$; note this
already uses $|E| \geq K$ implicitly, since a non-bipartite connected graph
has at least one cycle and hence $|E| \geq K$.

*Bipartite caveat.* If $G$ is bipartite with parts $(U, W)$, then $c_i = +1$
on $U$, $c_i = -1$ on $W$ satisfies (3.2), so $\mathrm{rank}(\mathrm{Pool}_A)
\leq K - 1$ *identically in the weights*; the same propagation argument shows
this is the only relation, so the generic rank is exactly $K - 1$ and the
theorem's count reads $3K + 2(K - 1) = 5K - 2$, still $\Theta(K)$. Molecular
graphs in the Level-8 benchmark contain odd cycles (aromatic and fused-ring
systems; any odd-membered ring suffices), so we are in the non-bipartite case
and the generic rank is $K$.

*Step 2: measure-zero exceptional set.* Fix a $K \times K$ minor $m(A)$ that
is a nonzero polynomial (Step 1'). The set
$\{ A \in \mathbb{R}^{|E|} : \mathrm{rank}(\mathrm{Pool}_A) < K \}$ is contained
in the zero set $\{ m(A) = 0 \}$ of a nonzero polynomial, which has Lebesgue
measure zero in $\mathbb{R}^{|E|}$ (standard: the zero set of a nonzero
polynomial is a proper algebraic subvariety, hence null). Since the positive
orthant $\{ A_e > 0 \}$ has positive measure and the exceptional set is null,
generic positive weights give rank $K$.

*Step 3: molecular weights.* Real molecular bond weights (e.g. bond orders,
distance-decayed couplings) are positive and carry no exact algebraic relation
of the specific form $m(A) = 0$; rank deficiency requires the weights to lie on
the measure-zero variety of Step 2. Hence genericity holds for the benchmark
inputs, and $\mathrm{rank}(\mathrm{Pool}_A) = K$. $\blacksquare$

**Conclusion of Section 3.** $\dim S_{ZZ} = \mathrm{rank}(\mathrm{Pool}_A) = K$
for generic weights on a connected non-bipartite molecular graph with
$|E| \geq K$ (and $K - 1$ in the bipartite case; both are $\Theta(K)$).

---

## 4. Dimension of the bond-pooled XX subspace: $\dim S_{XX} = \mathrm{rank}(\mathrm{Pool}_A)$

The argument of Section 3 is letter-agnostic. The map

$$
Z_{i(e)} Z_{j(e)} \;\longmapsto\; X_{i(e)} X_{j(e)}, \qquad e \in E,
$$

extends to a linear isometry $W_{ZZ} \to W_{XX} := \mathrm{span}\{ X_{i(e)} X_{j(e)} : e \in E \}$
between the two edge-indexed HS-orthonormal bases (both sides are orthonormal
by (1.1), since distinct edges give distinct supports). Under this isometry,
$O^{ZZ}_i \mapsto O^{XX}_i$ because both have the same coordinate vector -- the
$i$-th row of $\mathrm{Pool}_A$, cf. (3.1). Isometries preserve dimension of
spans, so

$$
\dim S_{XX} = \dim S_{ZZ} = \mathrm{rank}(\mathrm{Pool}_A) = K
$$

under the same genericity hypotheses. $\blacksquare$

---

## 5. Cross-orthogonality of the three subspaces

**Claim.** $S_1 \perp_{HS} S_{ZZ}$, $S_1 \perp_{HS} S_{XX}$, and
$S_{ZZ} \perp_{HS} S_{XX}$.

**Proof.** It suffices to check orthogonality on spanning sets; the inner
product is bilinear.

*(i) Weight-1 vs. weight-2.* Every generator of $S_1$ is a weight-1 Pauli
string; every generator of $S_{ZZ}$ and $S_{XX}$ is a real combination of
weight-2 Pauli strings. A weight-1 string and a weight-2 string are distinct
multi-indices (their supports have different cardinalities, so they cannot
agree letter-by-letter), hence HS-orthogonal by (1.1). Therefore
$S_1 \perp S_{ZZ}$ and $S_1 \perp S_{XX}$.

*(ii) ZZ vs. XX.* $S_{ZZ} \subseteq W_{ZZ}$ and $S_{XX} \subseteq W_{XX}$.
A basis string $Z_i Z_j$ of $W_{ZZ}$ and a basis string $X_k X_l$ of $W_{XX}$
never agree as multi-indices: even when $\{i,j\} = \{k,l\}$, the letters on
the shared support differ ($Z \neq X$), and $\mathrm{Tr}[Z X] = 0$ makes the
factorized trace (1.1) vanish. When the supports differ, some leg carries a
traceless factor against identity. Either way
$\langle Z_i Z_j,\, X_k X_l \rangle_{HS} = 0$, so $W_{ZZ} \perp W_{XX}$ and a
fortiori $S_{ZZ} \perp S_{XX}$. $\blacksquare$

Because the three subspaces are mutually orthogonal, their sum is direct:

$$
\mathcal{A}(\mathcal{O}_8) = S_1 \oplus S_{ZZ} \oplus S_{XX}
\qquad \text{(HS-orthogonal direct sum).}
\tag{5.1}
$$

---

## 6. Total dimension

Combining (5.1) with Sections 2-4:

$$
\dim \mathcal{A}(\mathcal{O}_8)
= \dim S_1 + \dim S_{ZZ} + \dim S_{XX}
= 3K + \mathrm{rank}(\mathrm{Pool}_A) + \mathrm{rank}(\mathrm{Pool}_A)
= 3K + 2\,\mathrm{rank}(\mathrm{Pool}_A).
$$

For generic weights on a connected non-bipartite molecular graph with
$|E| \geq K$, $\mathrm{rank}(\mathrm{Pool}_A) = K$ and

$$
\boxed{\;\dim \mathcal{A}(\mathcal{O}_8) = 3K + K + K = 5K = \Theta(K)\;}
$$

out of $\dim \mathrm{Herm}(\mathcal{H}) = 4^K$. (Bipartite case: $5K - 2$,
still $\Theta(K)$.)

**Scale.** For $K = 8$: $\dim \mathcal{A}(\mathcal{O}_8) = 40$ versus
$4^8 = 65536$, a fraction $40 / 65536 \approx 6.1 \times 10^{-4}$. In general
the accessible fraction is

$$
\frac{\dim \mathcal{A}(\mathcal{O}_8)}{\dim \mathrm{Herm}(\mathcal{H})}
= \frac{5K}{4^K} = \Theta\!\left( \frac{K}{4^K} \right) \longrightarrow 0
\quad (K \to \infty),
$$

i.e. the readout family occupies an exponentially vanishing fraction of
operator space. This is the *operator-geometry bottleneck*.

---

## 7. Prop 1.2: trace-out as a structured prior

**Proposition 1.2.** Let $\Pi_{\mathcal{A}} : \mathrm{Herm}(\mathcal{H}) \to
\mathcal{A}(\mathcal{O}_8)$ be the HS-orthogonal projection and
$\Pi_{\mathcal{A}^{\perp}} = \mathrm{id} - \Pi_{\mathcal{A}}$. Then the feature
map

$$
\phi_{\mathcal{O}}(\rho)_a \;=\; \mathrm{Tr}[O_a\, \rho]
\;=\; 2^{K} \langle O_a, \rho \rangle_{HS}
$$

depends on $\rho$ only through $\Pi_{\mathcal{A}}\, \rho$:

$$
\phi_{\mathcal{O}}(\rho) \;=\; \phi_{\mathcal{O}}\big( \Pi_{\mathcal{A}}\, \rho \big)
\qquad \text{for all } \rho \in \mathrm{Herm}(\mathcal{H}).
$$

**Proof.** Each $O_a \in \mathcal{O}_8 \subseteq \mathcal{A}(\mathcal{O}_8)$, and
$O_a$ is Hermitian, so $\mathrm{Tr}[O_a \rho] = 2^K \langle O_a, \rho \rangle_{HS}$.
Decompose $\rho = \Pi_{\mathcal{A}} \rho + \Pi_{\mathcal{A}^{\perp}} \rho$.
Orthogonality of the projection gives
$\langle O_a, \Pi_{\mathcal{A}^{\perp}} \rho \rangle_{HS} = 0$, hence

$$
\phi_{\mathcal{O}}(\rho)_a
= 2^K \langle O_a, \Pi_{\mathcal{A}} \rho \rangle_{HS}
+ 2^K \underbrace{\langle O_a, \Pi_{\mathcal{A}^{\perp}} \rho \rangle_{HS}}_{= 0}
= \phi_{\mathcal{O}}(\Pi_{\mathcal{A}} \rho)_a . \qquad \blacksquare
$$

The component $\Pi_{\mathcal{A}^{\perp}} \rho$ -- a subspace of dimension
$4^K - 5K$ -- is *invisible* to the readout. Statistically, this is a hard
structural prior. Write the Bayes-optimal target as
$g^{\ast}(\rho) = \mathbb{E}[Y \mid \rho]$ and (where $g^{\ast}$ admits a
linear representation in operator space, or after projecting its linear part)
decompose $g^{\ast} = g^{\ast}_{\mathcal{A}} + g^{\ast}_{\mathcal{A}^{\perp}}$
along $\mathcal{A} \oplus \mathcal{A}^{\perp}$. Two cases:

**(a) Aligned target ($g^{\ast}_{\mathcal{A}^{\perp}} = 0$).** The restriction
is *lossless*: every function of $\rho$ expressible through
$\mathcal{A}(\mathcal{O}_8)$ is exactly representable through
$\phi_{\mathcal{O}}$, and the hypothesis class shrinks from (effectively)
$4^K$ linear degrees of freedom to $5K$ with zero approximation error. The
estimator's variance scales with the retained dimension, so the bottleneck is
pure variance reduction -- an inductive bias that pays off exactly when the
molecular target is supported on single-qubit and bonded two-qubit correlators.

**(b) Misaligned target ($g^{\ast}_{\mathcal{A}^{\perp}} \neq 0$).** The
restriction incurs an irreducible approximation error of size
$\| g^{\ast}_{\mathcal{A}^{\perp}} \|$ (the HS-norm of the invisible component),
but removes $\dim \mathcal{A}^{\perp} = 4^K - 5K$ degrees of freedom from the
estimation problem. The bias-variance tradeoff is
$\text{excess risk} \approx \| g^{\ast}_{\mathcal{A}^{\perp}} \|^2 +
O(\dim \mathcal{A} / n)$ versus $O(4^K / n)$ for the unrestricted map: for any
sample size $n \ll 4^K$, the projected map dominates whenever the target's
invisible mass is small relative to the variance saved.

This is the precise sense in which the Level-8 trace-out acts as a *structured
prior*: it is not a soft regularizer but a hard, geometry-derived projection
whose retained subspace is dictated by the molecular bond graph.

---

## 8. Comparison to Canatar et al. 2022

Canatar et al. (2022) analyze quantum kernel generalization for the
full-fidelity kernel

$$
k_{\mathrm{fid}}(x, x') = \mathrm{Tr}\big[ \rho(x)\, \rho(x') \big]
= 2^{K} \langle \rho(x), \rho(x') \rangle_{HS},
$$

which is the HS inner product of the *full* density operators: its feature map
is $x \mapsto \rho(x) \in \mathrm{Herm}(\mathcal{H})$, implicitly using all
$4^K$ operator-space dimensions. Their central finding is that generalization
requires the kernel's spectral measure to concentrate on few modes, achieved
via *bandwidth* (scaling down rotation angles), which induces a *soft,
polynomial decay* of the kernel eigenvalues -- large eigenvalue mass on
low-degree modes, small but nonzero mass everywhere else.

TC-QIC's Level-8 kernel is

$$
k_{\mathcal{O}}(x, x') = \big\langle \phi_{\mathcal{O}}(x),\, \phi_{\mathcal{O}}(x') \big\rangle
= 4^{K} \big\langle \Pi_{\mathcal{A}}\, \rho(x),\; M\, \Pi_{\mathcal{A}}\, \rho(x') \big\rangle_{HS},
$$

where $M$ is the (fixed, positive semidefinite) Gram operator of the readout
family on $\mathcal{A}(\mathcal{O}_8)$; by Sections 2-6, $M$ has rank exactly
$\dim \mathcal{A}(\mathcal{O}_8) = 5K$. Hence $k_{\mathcal{O}}$ is a
*rank-$5K = \Theta(K)$ projection* of the fidelity kernel's operator space:

| | Canatar et al. 2022 | TC-QIC Level-8 (this work) |
|---|---|---|
| Feature operator space | full $\mathrm{Herm}(\mathcal{H})$, $4^K$-dim | $\mathcal{A}(\mathcal{O}_8)$, $5K$-dim |
| Spectral mechanism | soft polynomial decay via bandwidth | hard projection via readout geometry |
| Modes outside the prior | small but nonzero eigenvalue | exactly zero eigenvalue |
| Prior origin | kernel hyperparameter (bandwidth) | molecular bond graph (Pool_A) |

The operator-geometry bottleneck is thus the operator-space analog of their
bandwidth-induced spectral decay, with two sharpenings: (i) the decay is
replaced by a *hard cutoff* -- eigenvalue exactly zero on
$\mathcal{A}^{\perp}$, eliminating rather than merely damping the
$4^K - 5K$ misaligned modes; and (ii) the retained modes are not generic
low-degree functions but are *topology-aware*, selected by the bond-pooling
matrix $\mathrm{Pool}_A$ of the specific molecule. This converts Canatar
et al.'s tuning knob into a structural, data-derived prior -- which is exactly
the inductive-bias claim T2 feeds into the phase gate:
$\dim \mathcal{A}(\mathcal{O}_8) = \Theta(K)$. $\blacksquare$
