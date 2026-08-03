# T3: Spectral Coarse-Graining as an Ideal Low-Pass Filter

*Status: COMPLETE (honest scope) | Priority: MED | Deps: [] |*
*Unblocks: T7 (sufficiency chain), E3 (spectral probe)*

**Theorem (Lemma 3.5, informal).** Spectral coarse-graining of a molecular
graph into $K$ clusters, followed by cluster-mean pooling of node features, is an
*exact* orthogonal projection onto the bottom-$K$ Laplacian eigenspace *under the
continuous ratio-cut relaxation*, and an *approximate* such projection under the
`discretize` rounding that the implementation actually uses. The discarded
component is a well-defined high-graph-frequency band whose mutual information
with the target sets a topological ceiling on downstream AUC.

The honest scope statement, stated up front: the "ideal low-pass" property is a
theorem about the **relaxation**, not about the discrete label assignment. The
discretization contributes a controlled error $\varepsilon_{\mathrm{disc}}$ that
vanishes only in the large-spectral-gap limit.

---

## 1. Setup: the graph Laplacian and its spectrum

Let $G = (V, E, w)$ be a weighted molecular graph with $n$ atoms, symmetric
adjacency matrix $A \in \mathbb{R}^{n \times n}$ (bond orders as weights),
degree matrix $D = \mathrm{diag}\big(\sum_j A_{ij}\big)$, and normalized
Laplacian

$$
L_{\mathrm{sym}} \;=\; I - D^{-1/2} A\, D^{-1/2}.
$$

Its eigendecomposition is $L_{\mathrm{sym}} u_k = \lambda_k u_k$ with

$$
0 = \lambda_1 \le \lambda_2 \le \cdots \le \lambda_n \le 2 ,
\qquad u_k^{T} u_\ell = \delta_{k\ell}.
$$

For a node signal $f : V \to \mathbb{R}$ (equivalently $f \in \mathbb{R}^n$), the
**graph Fourier transform** is $\hat{f}(k) = u_k^{T} f$, so that
$f = \sum_k \hat{f}(k)\, u_k$. Low-frequency modes (small $\lambda_k$) capture
smooth, cluster-like variation; high-frequency modes (large $\lambda_k$) capture
node-to-node oscillation. The eigenvalue $\lambda_k$ *is* the graph frequency:
$u_k^{T} L_{\mathrm{sym}} u_k = \lambda_k$ measures the Dirichlet energy
(total edge-weighted variation) of the mode.

---

## 2. Spectral clustering as ratio-cut relaxation (Lemma 3.5)

**Lemma 3.5 (spectral clustering = relaxed ratio cut).** Partitioning $V$ into
$K$ disjoint clusters $C_1, \dots, C_K$ to minimize the normalized $K$-way
ratio cut

$$
\min_{C_1,\dots,C_K}\;
\sum_{k=1}^{K} \frac{\mathrm{cut}(C_k, C_k^{c})}{|C_k|},
\qquad
\mathrm{cut}(C_k, C_k^{c}) = \sum_{i \in C_k,\, j \notin C_k} A_{ij},
$$

is NP-hard as a discrete combinatorial problem. Its standard continuous
relaxation (von Luxburg 2007, Sec. 5) replaces the discrete indicator matrix by
an orthonormal continuous embedding and yields the closed-form solution

$$
U = [\,u_1, \dots, u_K\,] \in \mathbb{R}^{n \times K},
$$

the bottom-$K$ eigenvectors of $L_{\mathrm{sym}}$ (equivalently $L_{\mathrm{rw}}
= D^{-1}(D-A)$ up to the standard $D^{1/2}$ change of variables). The discrete
labels are then recovered by a rounding step applied to the rows of $U$: either
$k$-means (`assign_labels='kmeans'`) or the orthogonal-rounding
`assign_labels='discretize'` of Yu and Shi (2003), which is the variant used
here.

**Corollary (cluster-mean = subspace projection).** Let $f \in \mathbb{R}^n$ be
a node signal, and let the coarse feature for cluster $C_k$ be the cluster mean

$$
c_k \;=\; \frac{1}{|C_k|} \sum_{i \in C_k} f_i
\;=\; \frac{\mathbb{1}_{C_k}^{T} f}{\mathbb{1}_{C_k}^{T} \mathbb{1}_{C_k}},
$$

where $\mathbb{1}_{C_k} \in \{0,1\}^n$ is the indicator of cluster $C_k$. The map
$f \mapsto (c_1, \dots, c_K)$, lifted back to $\mathbb{R}^n$ by broadcasting each
$c_k$ across its cluster, is exactly the **orthogonal projection**

$$
P_{\mathcal{C}}\, f \;=\; \sum_{k=1}^{K}
\frac{\mathbb{1}_{C_k} \mathbb{1}_{C_k}^{T}}{|C_k|}\, f
$$

onto $\mathcal{C} = \mathrm{span}\{\mathbb{1}_{C_1}, \dots, \mathbb{1}_{C_K}\}$,
the space of piecewise-constant (cluster-constant) signals. Since the
$\mathbb{1}_{C_k}$ are disjointly supported, they are orthogonal and
$P_{\mathcal{C}}$ is a genuine rank-$K$ orthoprojector.

The relaxation identity $\mathrm{span}\{\mathbb{1}_{C_k}\} \approx
\mathrm{span}\{u_1, \dots, u_K\}$ is precisely what makes $P_{\mathcal{C}}$ an
**approximate low-pass filter**: it retains the component of $f$ in the
bottom-$K$ Laplacian eigenspace and discards the rest.

---

## 3. Formal statement (ideal relaxation, approximate discretization)

Let $\Pi_K = U(U^{T} U)^{-1} U^{T} = U U^{T}$ (the last equality since $U$ has
orthonormal columns) be the orthogonal projection onto the bottom-$K$ Laplacian
eigenspace $\mathrm{span}\{u_1, \dots, u_K\}$.

**Theorem (Lemma 3.5, full statement).**

**(a) [RELAXATION -- EXACT].** Under the continuous spectral embedding (before
rounding), the cluster-constant projector coincides with the spectral projector,

$$
P_{\mathcal{C}}^{\mathrm{relax}} \;=\; \Pi_K,
\qquad\text{so}\qquad
P_{\mathcal{C}}^{\mathrm{relax}} f \;=\; \sum_{k \le K} \hat{f}(k)\, u_k .
$$

All high-frequency content ($\lambda > \lambda_K$) is discarded exactly. This is
the **ideal low-pass filter**: an indicator function $\mathbb{1}[\lambda_k \le
\lambda_K]$ in the graph-frequency domain.

**(b) [DISCRETIZATION -- APPROXIMATE].** Under the `discretize` rounding (the
actual implementation), let $U_{\mathrm{disc}}$ be the rounded indicator
embedding and $P_{\mathcal{C}}$ the resulting cluster-mean operator. Then

$$
P_{\mathcal{C}} f \;=\; \Pi_K f + \varepsilon_{\mathrm{disc}},
\qquad
\|\varepsilon_{\mathrm{disc}}\| \;\le\; \delta\, \|f\|,
\qquad
\delta = O\!\big(\|U_{\mathrm{disc}} - U\|_2\big).
$$

By the Davis-Kahan $\sin\Theta$ theorem, the subspace perturbation is controlled
by the spectral gap:

$$
\|U_{\mathrm{disc}} - U\|_2 \;\lesssim\;
\frac{\|E\|_2}{\lambda_{K+1} - \lambda_K},
$$

where $\|E\|_2$ is the effective rounding perturbation. Hence $\delta$ is small
when the gap $\lambda_{K+1} - \lambda_K$ is large (well-separated clusters), and
grows as the gap closes. In the well-clustered limit $\lambda_{K+1} - \lambda_K
\to$ (constant) $> 0$ and $\|E\|_2 \to 0$, so $\delta \to 0$ and (b) collapses to
(a).

**Honest scope.** The implementation runs `SpectralClustering` on the bond
adjacency $A$ and pools by discrete labels, i.e. it computes $P_{\mathcal{C}}$ of
part (b), not $\Pi_K$ of part (a). The "ideal low-pass" claim is exact only for
the relaxation; for the shipped code it holds up to $\varepsilon_{\mathrm{disc}}$,
which is provably small only when the spectral gap is open. No stronger claim is
made.

---

## 4. "High-frequency atomic noise" (Definition 3.4)

**Definition 3.4 (high-frequency atomic noise).** The *high-frequency atomic
noise* of a molecular node signal $f$ at coarsening level $K$ is the discarded
residual

$$
f_{\mathrm{HF}} \;=\; f - \Pi_K f \;=\; \sum_{k > K} \hat{f}(k)\, u_k .
$$

This is the within-cluster atomic variation stripped away by coarse-graining. Its
graph-frequency content is pure high band, $\lambda > \lambda_K$: by construction
$\Pi_K f_{\mathrm{HF}} = 0$ and $u_k^{T} f_{\mathrm{HF}} = 0$ for all $k \le K$.

**Remark.** The coarse qubit features $\mathrm{qf}$ (atomic number, Gasteiger
charge, degree, aromaticity, ring membership -- the five columns of `feats` in
`coarse_graph`) are node signals on the full molecular graph. After coarsening,

$$
\mathrm{qf}_{\mathrm{coarse}} \;=\; \Pi_K\, \mathrm{qf} + \varepsilon_{\mathrm{disc}},
$$

column by column. The high-frequency component $\mathrm{qf}_{\mathrm{HF}} =
\mathrm{qf} - \Pi_K\, \mathrm{qf}$ is precisely the within-cluster variation
probed by E10 (differential feature-injection test): injecting
$\mathrm{qf}_{\mathrm{HF}}$ back into the head measures how much of the lost band
the target actually needed.

---

## 5. Topological bottleneck statement

**Corollary 3.6 (topological information bottleneck).** Spectral coarsening to
$K$ clusters discards mutual information bounded by the high-frequency band:

$$
I(f_{\mathrm{HF}}; Y) \;\le\; I(f; Y) - I(\Pi_K f; Y),
$$

the mutual information carried in modes $\lambda > \lambda_K$ (data-processing
inequality applied to $f \mapsto (\Pi_K f, f_{\mathrm{HF}})$).

- **Low-graph-frequency tasks** (toxicophore *location* = a cluster-level
  pattern): $I(\Pi_K f; Y) \approx I(f; Y)$, so the discarded information is
  small and coarsening is nearly lossless.
- **High-graph-frequency tasks** (atom-level distinctions *within* a ring):
  $I(\Pi_K f; Y) \ll I(f; Y)$, so the bottleneck is tight and the achievable
  AUC ceiling is low.

This is the **formal origin of the observed AUC ceiling** ($\approx 0.61$--$0.66$
on the Level-8 family): the topology $L = D - A$ suffices for most toxicity
endpoints, but within-cluster atomic detail is compressed into
$f_{\mathrm{HF}}$ and cannot be recovered downstream.

---

## 6. Implementation reconciliation

The actual code is

```python
SpectralClustering(n_clusters=k, affinity='precomputed',
                   assign_labels='discretize', random_state=0).fit_predict(A + 1e-6)
```

run on the **bond adjacency $A$** (not on $L$), with cluster-mean pooling
`qf[c] = feats[labels==c].mean(0)`. Reconciling this with Sections 2-3:

1. With `affinity='precomputed'`, scikit-learn treats $A$ as the affinity
   (similarity) matrix and internally forms the normalized Laplacian
   $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$, then embeds using its **bottom-$K$
   eigenvectors**. So the eigenvectors used are those of $L$, not of $A$ directly.

2. Eigenvalues of $A$ and $L_{\mathrm{sym}}$ are related by
   $\lambda_L = 1 - \mu_A$, where $\mu_A$ are the eigenvalues of the normalized
   adjacency $D^{-1/2} A D^{-1/2} \in [-1, 1]$. Hence the **bottom-$K$
   eigenvectors of $L$** (smallest $\lambda_L$) are the **top-$K$ eigenvectors of
   the normalized adjacency** (largest $\mu_A$, most positive / most connected).

3. Therefore clustering with `affinity=`$A$ is equivalent, up to the fixed
   re-ordering $\lambda_L = 1 - \mu_A$, to coarsening onto the **top-$K$
   network-community modes** -- the large-positive-eigenvalue modes of $A$ that
   correspond to smooth, cluster-like structure. These *are* the low-graph-
   frequency Laplacian modes.

**Cross-reference convention.** Throughout Sections 2-5, "bottom-$K$ of $L$" and
"top-$K$ of the normalized $A$" denote the *same* subspace $\mathrm{span}\{u_1,
\dots, u_K\}$; the $\Pi_K$ projector is unchanged. The relation $\lambda_L = 1 -
\mu_A$ is the only bookkeeping needed, and it reorders modes without altering the
low-pass conclusion.

One further note: the pooled edge matrix `Ac` in the code aggregates only
*between-cluster* bond weights (the `labels[i] != labels[j]` guard), so the
coarse graph $C(G)$ retains inter-cluster topology while the intra-cluster
structure is exactly the discarded $f_{\mathrm{HF}}$ band -- consistent with
Definition 3.4.

---

## 7. Connection to TC-QIC

The spectral low-pass is the **first** bottleneck in the double-bottleneck
picture:

$$
\underbrace{G \;\longrightarrow\; C(G)}_{\text{topological bottleneck (T3)}}
\qquad
\underbrace{\rho \;\longrightarrow\; \phi_{\mathcal{O}_8}(\rho)}_{\text{operator bottleneck (T2)}}
$$

- **Topological bottleneck (T3):** molecular graph $G$ is compressed to the
  coarsened qubit graph $C(G)$ via $\Pi_K$ (plus $\varepsilon_{\mathrm{disc}}$).
- **Operator bottleneck (T2):** the qubit state $\rho$ is compressed to the
  measured Level-8 features $\phi_{\mathcal{O}_8}(\rho)$ via the rank-$5K$
  operator projection.

The two bottlenecks act **in series**: information must survive *both* to reach
the head. The compound compression obeys

$$
I(G; Y) \;\ge\; I\big(C(G); Y\big) \;\ge\; I\big(\phi_{\mathcal{O}_8}(\rho); Y\big),
$$

with the two inequalities becoming near-equalities exactly at the two measured
AUC ceilings (E10 for the topological stage, E4 for the operator stage).

---

## 8. E3 experiment (verification)

**Goal.** Empirically confirm that cluster-mean coarsening acts as a low-pass
filter, and quantify the discretization residual $\varepsilon_{\mathrm{disc}}$.

**Procedure.**

1. For each molecule, build $L_{\mathrm{sym}}$ from $A$ and compute the
   eigenbasis $\{u_k, \lambda_k\}$.
2. Take each coarse feature column $\mathrm{qf}_{\cdot}$ (broadcast back to
   $\mathbb{R}^n$) and compute its graph Fourier coefficients $\hat{f}(k)$.
3. Measure the **low-frequency energy fraction**

$$
\eta_K \;=\; \frac{\sum_{k \le K} \hat{f}(k)^2}{\sum_{k} \hat{f}(k)^2}
\;=\; \frac{\|\Pi_K f\|^2}{\|f\|^2},
$$

   and the residual $\|f_{\mathrm{HF}}\|^2 / \|f\|^2 = 1 - \eta_K$.
4. Report $\varepsilon_{\mathrm{disc}}$ as the gap between the discrete
   cluster-mean projection and $\Pi_K$.

**Pass criterion.** Low-frequency fraction $\eta_K > 0.7$ (features are mostly
smooth / cluster-constant), with the residual bounded in accordance with the
spectral gap $\Delta = \lambda_{K+1} - \lambda_K$: molecules with larger $\Delta$
should exhibit smaller $\varepsilon_{\mathrm{disc}}$, per Section 3(b).

**Script.** `probe_spectral_lowpass.py` (E3, to be written).

---

*Deps: none. Unblocks T7 (sufficiency chain via the series-bottleneck inequality
of Section 7) and E3 (spectral probe of Section 8).*
