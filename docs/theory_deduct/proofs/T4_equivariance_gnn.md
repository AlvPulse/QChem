# T4: Task Symmetry, $S_K$ Equivariance, and Quantum Message Passing

*Status: COMPLETE | Priority: HIGH | Deps: none*

---

## 0. Setup and notation

Let $G = (V, E)$ be a molecular graph on $n$ atoms with adjacency matrix
$A \in \mathbb{R}^{n \times n}$, $A_{ij} \ge 0$. A relabeling of atoms is a
permutation $\sigma \in S_n$ acting on $G$ by
$(\sigma \cdot G)$ with adjacency $A^{\sigma}_{ij} = A_{\sigma^{-1}(i),\, \sigma^{-1}(j)}$.
We write $[G] = \{\sigma \cdot G : \sigma \in S_n\}$ for the orbit (isomorphism
class) of $G$ under the $S_n$-action.

The quantum state produced by the circuit for graph $G$ is a density operator
$\rho$ on the $K$-qubit Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes K}$.
For each qubit index $i$ we fix a single-qubit observable
$O_i = I^{\otimes(i-1)} \otimes P \otimes I^{\otimes(K-i)}$, where $P$ is a fixed
single-qubit Pauli (the same $P$ on every wire). A qubit permutation
$\pi \in S_K$ is represented on $\mathcal{H}$ by the unitary $U_\pi$ that permutes
tensor factors:

$$
U_\pi \,(v_1 \otimes v_2 \otimes \cdots \otimes v_K)
   = v_{\pi^{-1}(1)} \otimes v_{\pi^{-1}(2)} \otimes \cdots \otimes v_{\pi^{-1}(K)}.
$$

The **bond-pooled readout** is the vector-valued map

$$
B_A(\rho)_i \;=\; \sum_{j} A_{ij}\, \mathrm{Tr}\!\left[O_i O_j\, \rho\right],
\qquad i = 1, \dots, K,
$$

and its **connected** variant uses the connected two-point correlator

$$
C_{ij}(\rho) \;=\; \mathrm{Tr}[O_i O_j\, \rho] \;-\; \mathrm{Tr}[O_i\, \rho]\,\mathrm{Tr}[O_j\, \rho].
$$

---

## 1. Label invariance and orbit sufficiency (Prop 3.3)

**Proposition 3.3 (Orbit sufficiency).**
The toxicity target $Y$ is invariant under atom relabeling,
$Y(\sigma \cdot G) = Y(G)$ for all $\sigma \in S_n$. Consequently the orbit $[G]$
is a sufficient statistic for $Y$: $p(Y \mid G) = p(Y \mid [G])$.

**Proof.**
Toxicity is a physical property of the molecule, not of the arbitrary index we
assign to each atom; permuting atom labels produces the same chemical object, so
$Y(\sigma \cdot G) = Y(G)$ by definition of the labeling being a gauge choice.

$Y$ is therefore constant on each orbit $[G]$. Write the orbit-projection map
$q: G \mapsto [G]$. Because $Y$ factors through $q$ (i.e. there exists $\tilde{Y}$
with $Y = \tilde{Y} \circ q$), the conditional law of $Y$ given $G$ depends on $G$
only through $[G]$:

$$
p(Y \mid G) \;=\; p\big(Y \mid q(G)\big) \;=\; p(Y \mid [G]).
$$

This is exactly the Fisher-Neyman factorization criterion: the likelihood
factors as $p(Y \mid G) = g\big(Y, [G]\big)\cdot h(G)$ with $h \equiv 1$, so $[G]$
is a sufficient statistic for $Y$. $\qquad\blacksquare$

**Consequence (capacity argument).**
Let $f$ be any hypothesis. Decompose $f = f_{\mathrm{inv}} + f_{\perp}$ into its
component that factors through $[G]$ and its orthogonal complement (the part that
distinguishes graphs within the same orbit). Since $Y$ is measurable with respect
to $\sigma([G])$, the component $f_{\perp}$ can only fit noise: it contributes to
variance without reducing the irreducible risk. Any architecture that does not
factor through $[G]$ spends representational capacity on distinctions the target
provably cannot see. An architecture whose outputs are orbit-measurable (i.e.
$S_n$-invariant after the graph-to-qubit embedding, realized below as
$S_K$-invariance) has this wasted component identically zero.

---

## 2. $S_K$ equivariance of the bond-pooled readout (Thm 2.4)

**Theorem 2.4 (Readout equivariance).**
For every qubit permutation $\pi \in S_K$,

$$
B_{A^\pi}\!\big(U_\pi \rho\, U_\pi^\dagger\big)_i
   \;=\; B_A(\rho)_{\pi^{-1}(i)},
$$

where $A^\pi_{ij} = A_{\pi^{-1}(i),\,\pi^{-1}(j)}$. In words: permuting the qubits
of the state and permuting the adjacency in the same way permutes the readout
vector by the same permutation. The map $B$ is $S_K$-equivariant.

**Lemma 2.4a (Pauli relabeling under conjugation).**
For single-qubit observables $O_i = I \otimes \cdots \otimes P \otimes \cdots \otimes I$
(Pauli $P$ on wire $i$) and any $\pi \in S_K$,

$$
U_\pi^\dagger\, O_i\, U_\pi \;=\; O_{\pi^{-1}(i)},
\qquad
U_\pi^\dagger\, (O_i O_j)\, U_\pi \;=\; O_{\pi^{-1}(i)}\, O_{\pi^{-1}(j)}.
$$

*Proof of Lemma.*
$U_\pi$ maps the tensor factor on wire $k$ to wire $\pi(k)$. Conjugation
$U_\pi^\dagger (\cdot) U_\pi$ therefore sends an operator acting nontrivially on
wire $i$ to one acting nontrivially on wire $\pi^{-1}(i)$, leaving the (identical)
Pauli letter $P$ unchanged and identities on all other wires. Explicitly, on a
product vector,

$$
U_\pi^\dagger O_i U_\pi \,(v_1 \otimes \cdots \otimes v_K)
= U_\pi^\dagger O_i (v_{\pi^{-1}(1)} \otimes \cdots \otimes v_{\pi^{-1}(K)}).
$$

$O_i$ applies $P$ to the factor sitting in slot $i$, which is $v_{\pi^{-1}(i)}$;
applying $U_\pi^\dagger$ returns each factor to its original slot, so $P$ now acts
on the slot originally holding $v_{\pi^{-1}(i)}$, i.e. on wire $\pi^{-1}(i)$. Hence
$U_\pi^\dagger O_i U_\pi = O_{\pi^{-1}(i)}$. Since conjugation is an algebra
homomorphism, $U_\pi^\dagger (O_i O_j) U_\pi = (U_\pi^\dagger O_i U_\pi)(U_\pi^\dagger O_j U_\pi)
= O_{\pi^{-1}(i)} O_{\pi^{-1}(j)}$. $\qquad\square$

**Proof of Theorem 2.4.**
Start from the definition and substitute $A^\pi$:

$$
\mathrm{LHS}_i
= B_{A^\pi}\big(U_\pi \rho U_\pi^\dagger\big)_i
= \sum_j A^\pi_{ij}\, \mathrm{Tr}\!\left[O_i O_j\, U_\pi \rho U_\pi^\dagger\right]
= \sum_j A_{\pi^{-1}(i),\,\pi^{-1}(j)}\, \mathrm{Tr}\!\left[O_i O_j\, U_\pi \rho U_\pi^\dagger\right].
$$

Use cyclicity of the trace and $U_\pi^\dagger U_\pi = I$:

$$
\mathrm{Tr}\!\left[O_i O_j\, U_\pi \rho U_\pi^\dagger\right]
= \mathrm{Tr}\!\left[U_\pi^\dagger O_i O_j\, U_\pi \,\rho\right]
= \mathrm{Tr}\!\left[O_{\pi^{-1}(i)} O_{\pi^{-1}(j)}\, \rho\right],
$$

where the last equality is Lemma 2.4a. Therefore

$$
\mathrm{LHS}_i
= \sum_j A_{\pi^{-1}(i),\,\pi^{-1}(j)}\, \mathrm{Tr}\!\left[O_{\pi^{-1}(i)} O_{\pi^{-1}(j)}\, \rho\right].
$$

Set $k = \pi^{-1}(i)$ and reindex the sum by $l = \pi^{-1}(j)$. As $j$ ranges over
$\{1,\dots,K\}$ so does $l$ (bijection), giving

$$
\mathrm{LHS}_i
= \sum_l A_{k,\,l}\, \mathrm{Tr}\!\left[O_k O_l\, \rho\right]
= B_A(\rho)_k
= B_A(\rho)_{\pi^{-1}(i)}
= \mathrm{RHS}_i. \qquad\blacksquare
$$

**Remark.** The identical argument applies verbatim to the connected readout
$b_i = \sum_j A_{ij} C_{ij}(\rho)$, because each of the three trace factors in
$C_{ij}$ transforms the same way under conjugation:
$\mathrm{Tr}[O_i \rho] \to \mathrm{Tr}[O_{\pi^{-1}(i)}\rho]$ etc., so $C_{ij}$ inherits
the same index relabeling. Hence the connected bond-pooled readout is likewise
$S_K$-equivariant.

**Corollary 2.4b (Invariant global head).**
The pooled scalar $\bar{B}(\rho) = \tfrac{1}{K}\sum_i B_A(\rho)_i$ is
$S_K$-invariant:

$$
\bar{B}\big(U_\pi \rho U_\pi^\dagger;\, A^\pi\big)
= \tfrac{1}{K}\sum_i B_{A^\pi}(U_\pi \rho U_\pi^\dagger)_i
= \tfrac{1}{K}\sum_i B_A(\rho)_{\pi^{-1}(i)}
= \tfrac{1}{K}\sum_k B_A(\rho)_k
= \bar{B}(\rho; A),
$$

using that $i \mapsto \pi^{-1}(i)$ is a bijection so the sum is unchanged. The
global classifier thus receives a permutation-invariant input, and by the
graph-to-qubit correspondence this realizes the orbit-measurability demanded by
Prop 3.3. $\qquad\blacksquare$

---

## 3. Equivariant hypothesis class and Rademacher tightening (Cor 2.5)

By Theorem 2.4 the readout is equivariant under the full $S_K$, hence in
particular under the automorphism group $\mathrm{Aut}(G) \le S_K$ (those
permutations fixing $A$, i.e. $A^\pi = A$). The induced hypothesis class

$$
\mathcal{H}_{O_8} \;\subseteq\; \mathcal{F}_{\mathrm{equiv}}(G)
= \big\{ f : f(U_\pi \rho U_\pi^\dagger) = \Pi_\pi\, f(\rho)\ \ \forall\, \pi \in \mathrm{Aut}(G) \big\},
$$

where $\Pi_\pi$ is the corresponding output permutation (identity after the
invariant pooling of Cor 2.4b).

**Corollary 2.5 (Complexity tightening).**
For an equivariant class under a finite group $\Gamma = \mathrm{Aut}(G)$, the
empirical Rademacher complexity satisfies (Elesedy and Zaidi, 2021)

$$
\mathfrak{R}_m\big(\mathcal{F}_{\mathrm{equiv}}\big)
\;\le\; |\Gamma|^{-1/2}\; \mathfrak{R}_m\big(\mathcal{F}\big),
$$

so the generalization bound improves by a factor $|\mathrm{Aut}(G)|^{-1/2}$
relative to the unconstrained class $\mathcal{F}$.

**Honest quantitative statement.** For most drug-like molecules $\mathrm{Aut}(G)$
is trivial or near-trivial ($|\mathrm{Aut}(G)| \in \{1, 2\}$), so the numeric
factor $|\mathrm{Aut}(G)|^{-1/2} \in \{1,\, 0.707\}$ is modest and often exactly
$1$. The load-bearing benefit is qualitative rather than numeric: the model
structurally cannot expend capacity on physically equivalent relabelings (the
$f_\perp$ component of Prop 3.3 is identically zero), which removes a source of
variance regardless of how large $|\mathrm{Aut}(G)|$ happens to be.

---

## 4. Quantum message-passing identity (Prop 2.6)

**Proposition 2.6 (Readout is one MPNN layer).**
The connected bond-pooled readout

$$
b_i \;=\; \sum_j A_{ij}\, C_{ij}(\rho),
\qquad
C_{ij}(\rho) = \mathrm{Tr}[O_i O_j \rho] - \mathrm{Tr}[O_i \rho]\,\mathrm{Tr}[O_j \rho],
$$

is exactly one step of a message-passing graph neural network in the
Gilmer et al. (2017) form

$$
h_i' = \phi\Big(h_i,\; \textstyle\sum_{j \in N(i)} \psi(h_i, h_j, A_{ij})\Big),
$$

under the identification

- node feature: $\quad h_i = \mathrm{Tr}[O_i \rho]$ (single-qubit expectation);
- message:      $\quad \psi(i, j) = A_{ij}\, C_{ij}(\rho)$ (bond-weighted connected correlator);
- aggregation:  $\quad \bigoplus = \sum$ over neighbors (fixed, not learned);
- update:       $\quad \phi = \mathrm{id}$, so $h_i' = b_i$ is the new node state.

**Proof.**
Substitute the identifications into the Gilmer template. The neighbor set is
$N(i) = \{ j : A_{ij} \neq 0 \}$, so

$$
\sum_{j \in N(i)} \psi(h_i, h_j, A_{ij})
= \sum_{j \in N(i)} A_{ij}\, C_{ij}(\rho)
= \sum_{j} A_{ij}\, C_{ij}(\rho),
$$

the last equality because terms with $A_{ij} = 0$ vanish. Applying
$\phi = \mathrm{id}$ (drop the residual $h_i$ slot, or absorb it as a
$0$-weight self term) gives $h_i' = \sum_j A_{ij} C_{ij}(\rho) = b_i$. This is
precisely the readout. $\qquad\blacksquare$

**Difference from a classical MPNN.**
In a classical MPNN the message $\psi(h_i, h_j, A_{ij})$ is a *learned function of
single-node features* $h_i, h_j$. Here the message contains the genuine two-qubit
connected correlator $C_{ij}(\rho)$, which in general is **not** a function of the
single-qubit marginals $h_i = \mathrm{Tr}[O_i\rho]$, $h_j = \mathrm{Tr}[O_j\rho]$:

$$
C_{ij}(\rho) \neq 0 \quad\text{even when}\quad h_i, h_j\ \text{are fixed},
$$

precisely when $\rho$ carries $i$-$j$ correlations (entanglement or classical
correlation) beyond the product of marginals. A classical MPNN whose messages are
built only from node features $h_i, h_j$ cannot reproduce $C_{ij}$ in general; the
quantum readout injects a two-body signal that no single-qubit observable can
encode. This is the concrete sense in which the message channel carries an
entanglement signal.

---

## 5. Relation to Xu et al. (2019) WL expressivity

Xu et al. (2019, GIN) prove that **sum** aggregation is maximally discriminative
among neighborhood aggregators: unlike mean or max, an injective sum-based
aggregator can match the discriminative power of the Weisfeiler-Lehman (WL) graph
isomorphism test, whereas mean and max provably collapse distinct multisets.

The bond-pooled readout uses sum aggregation weighted by $A_{ij}$
(Prop 2.6), placing it in the maximally-discriminative regime for its
neighborhood-aggregation step. Combined with Prop 3.3 (orbit sufficiency) and
Thm 2.4 (equivariance), this yields:

> **TC-QIC's readout is WL-aligned.** It is as expressive as the graph
> coarse-graining permits (sum aggregation, Xu 2019), and it respects the label
> symmetry of the target (orbit sufficiency, Prop 3.3; equivariance, Thm 2.4).

The readout therefore neither under-reaches (it retains WL-level discriminative
power through sum pooling) nor over-reaches (it factors through the orbit and
cannot distinguish label-equivalent graphs), matching expressivity to the
symmetry structure of the toxicity task.

---

## References

- J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, G. E. Dahl.
  *Neural Message Passing for Quantum Chemistry.* ICML 2017.
- K. Xu, W. Hu, J. Leskovec, S. Jegelka.
  *How Powerful are Graph Neural Networks?* (GIN.) ICLR 2019.
- B. Elesedy, S. Zaidi.
  *Provably Strict Generalisation Benefit for Equivariant Models.* ICML 2021.
