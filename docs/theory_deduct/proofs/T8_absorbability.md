# T8: The Absorbability Theorem
*Status: COMPLETE | Priority: HIGH | Phase-gate: Cor 2.8*

## 1. Formal Setup

We consider a parameterized model that reads a structured feature vector $h \in
\mathbb{R}^d$ (e.g. a chemistry-derived or topology-derived descriptor of a
molecule) at one or more *encoding sites* of a circuit or network.

Definitions:

- **Encoding site $s$**: a location in the model where a scalar rotation angle
  (or gate parameter) is produced from the feature $h$. Each site $s$ emits an
  angle $\theta_s$.
- **Free trainable row $w_s \in \mathbb{R}^d$**: the parameters of a single
  linear map feeding site $s$. When every site has its own independent row we
  collect them into a trainable matrix $W = [w_1; w_2; \ldots; w_S] \in
  \mathbb{R}^{S \times d}$, so the stacked angle vector is $\theta = W h$.
- **Scramble permutation $\pi_s$**: a fixed permutation of the $d$ coordinates of
  $h$ applied at site $s$ *before* the trainable map. We write $P_{\pi_s}$ for the
  corresponding $d \times d$ permutation matrix ($P_{\pi_s}^{-1} = P_{\pi_s}^{\top}$).
  In the "structured" model $\pi_s = \mathrm{id}$; in the "scrambled" model
  $\pi_s$ is a nontrivial fixed permutation.
- **Structured class $\mathcal{H}_{\mathrm{struct}}$**: the set of input-output
  functions realizable as the trainable parameters range over their domain, with
  every $\pi_s = \mathrm{id}$.
- **Scrambled class $\mathcal{H}_{\mathrm{scram}}$**: the same set of functions but
  with the fixed permutations $\pi_s$ inserted at each site.

Let $R(\cdot)$ denote the (population or empirical) risk functional on functions.
"$\mathcal{H}_{\mathrm{struct}} = \mathcal{H}_{\mathrm{scram}}$ as function classes"
means the two families realize exactly the same set of functions, hence
$\inf_{f \in \mathcal{H}_{\mathrm{struct}}} R(f) = \inf_{f \in \mathcal{H}_{\mathrm{scram}}} R(f)$
for every $R$. Any nonzero measured *gap* between the two is then attributable to
optimization or finite-sample effects, not to representational content.

---

## 2. Theorem Statement (Condition A: single permutation)

**Theorem (A).** Suppose each site $s$ reads $h$ through its *own* free projection
$w_s$, and a *single* fixed permutation $\pi_s$ is applied to $h$ at that site
before the projection. Then
$$
\mathcal{H}_{\mathrm{struct}} = \mathcal{H}_{\mathrm{scram}}
$$
as function classes.

**Proof.** In the scrambled model the angle at site $s$ is
$$
\theta_s^{\mathrm{scram}} = w_s^{\top} P_{\pi_s} h = (P_{\pi_s}^{\top} w_s)^{\top} h .
$$
Define the reparametrization
$$
w_s' := P_{\pi_s}^{\top} w_s = P_{\pi_s}^{-1} w_s ,
\qquad\text{equivalently}\qquad
W' := \operatorname{diag\text{-}rows}(P_{\pi_s}^{-1})\,W ,
$$
i.e. each trainable row is permuted by that site's inverse permutation. Then
$$
\theta_s^{\mathrm{scram}} = (w_s')^{\top} h = \theta_s^{\mathrm{struct}} ,
$$
so the scrambled model with parameters $\{w_s\}$ computes exactly the same angle
vector as the structured model with parameters $\{w_s'\}$.

The map $W \mapsto W'$ (permute the coordinates of each row by $P_{\pi_s}^{-1}$) is
a linear bijection of the trainable parameter space onto itself: it is invertible
with inverse $W' \mapsto \operatorname{diag\text{-}rows}(P_{\pi_s})\,W'$, and it
carries the full domain onto the full domain (permutation of coordinates preserves
any coordinate-symmetric parameter domain such as $\mathbb{R}^{S\times d}$ or an
$\ell_p$ ball). Because everything downstream of the angles is identical in the two
models, and the angle-generating map is the same function of $(W, h)$ up to this
bijection of $W$, the two families realize the *same* set of functions:
$$
\mathcal{H}_{\mathrm{scram}}
= \{\, f_{W} : W \,\}
= \{\, f_{W'} : W' \,\}
= \mathcal{H}_{\mathrm{struct}} .
$$

Consequently
$$
\inf_{\mathcal{H}_{\mathrm{scram}}} R
= \inf_{\mathcal{H}_{\mathrm{struct}}} R
\quad\text{for every risk } R .
$$
The risk gap is exactly zero. Any gap observed empirically is optimization or
finite-sample noise. $\qquad\blacksquare$

**Remark.** The single free projection per site is what makes the permutation a
mere relabeling of trainable weights. The permutation moves *which* input
coordinate multiplies *which* weight, but since every weight is free and the
correspondence is a bijection, the reachable set of linear functionals is
unchanged.

---

## 3. Theorem Statement (Condition B: fixed per-molecule data)

**Theorem (B).** Suppose the structure enters not through the trainable projection
but as a *fixed per-molecule datum* $A(\mathrm{mol})$ that varies with the input
molecule and multiplies a physical observable, and that this multiplication happens
*upstream of all trainable layers*. Then
$$
\mathcal{H}_{\mathrm{struct}} \neq \mathcal{H}_{\mathrm{scram}} .
$$

**Proof.** Absorbability in Condition A relied on the existence of a *single fixed*
$W'$ that reproduces the scrambled computation for *all* inputs simultaneously.
Here the structured quantity $A(\mathrm{mol})$ enters the state/observable *before*
any trainable weight, so to absorb a scramble of $A$ into the weights we would need
a fixed $W'$ satisfying
$$
W' h = f\big(A(\mathrm{mol})\big) \qquad \text{for every molecule } \mathrm{mol}
$$
where the right-hand side is the effect that the structured (unscrambled) data has.
But $A(\mathrm{mol})$ takes different values across molecules while $W'$ is a single
fixed matrix and $h$ is the (possibly molecule-independent, or independently
supplied) feature. A fixed linear map cannot equal a molecule-dependent target for
all molecules at once unless that target were already a fixed linear function of the
available features -- which it is not, because $A(\mathrm{mol})$ has been injected
upstream and mixed with the physical observable in a way no downstream linear map
can undo per-input.

Formally: pick two molecules $\mathrm{mol}_1 \neq \mathrm{mol}_2$ with
$A(\mathrm{mol}_1) \neq A(\mathrm{mol}_2)$ but identical downstream trainable
inputs $h_1 = h_2$. Absorption would require $W' h_1 = f(A(\mathrm{mol}_1))$ and
$W' h_2 = f(A(\mathrm{mol}_2))$ with $h_1=h_2$, forcing
$f(A(\mathrm{mol}_1)) = f(A(\mathrm{mol}_2))$, a contradiction. Hence no single
$W'$ absorbs the structure; the two function classes are genuinely distinct and the
risk gap is meaningful. $\qquad\blacksquare$

**Remark.** The essential asymmetry: in Condition A the structure is a *permutation
of the trainable input*, which the trainable weights can re-permute. In Condition B
the structure is a *per-input multiplicative modulation of a physical observable*
placed before any weight, so it changes the reachable function set itself.

---

## 4. Non-absorbability via multiple inconsistent permutations

**Theorem (C).** Suppose the *same* projected vector $Wh$ is fed to two or more
sites, and site $s_1$ applies permutation $\pi_{s_1}$ while site $s_2$ applies a
*different* permutation $\pi_{s_2}$, with the projection $W$ *shared* across those
sites (not an independent free row per site). Then the reparametrization argument of
Condition A fails, and $\mathcal{H}_{\mathrm{struct}} \neq \mathcal{H}_{\mathrm{scram}}$
(partial non-absorbability).

**Proof.** With a shared $W$, absorbing the scramble at site $s_1$ requires
$W' = P_{\pi_{s_1}}^{-1} W$, while absorbing it at site $s_2$ requires
$W' = P_{\pi_{s_2}}^{-1} W$. A single $W'$ must satisfy both:
$$
P_{\pi_{s_1}}^{-1} W = P_{\pi_{s_2}}^{-1} W
\quad\Longleftrightarrow\quad
(P_{\pi_{s_2}} P_{\pi_{s_1}}^{-1} - I)\,W = 0 .
$$
For generic $W$ (full support rows) this forces
$P_{\pi_{s_1}} = P_{\pi_{s_2}}$, i.e. $\pi_{s_1} = \pi_{s_2}$. When the permutations
are inconsistent ($\pi_{s_1} \neq \pi_{s_2}$) the system is overdetermined and no
single shared $W'$ reproduces both sites. The scramble therefore cannot be fully
reparametrized away; the function classes differ. The degree of non-absorbability
grows with the number of mutually inconsistent permutations sharing a projection
(Levels 3, 5, 6, 7). $\qquad\blacksquare$

**Remark.** With $k$ mutually inconsistent permutations sharing one projection, the
absorbing constraint is a system of $k-1$ independent equations
$(P_{\pi_{s_j}} P_{\pi_{s_1}}^{-1} - I)W = 0$; the more constraints, the smaller the
absorbable subspace, hence the more genuine the gap.

---

## 5. Pre-registration criterion (Corollary 2.8)

**Corollary 2.8 (Pre-registration test).** For any structured-vs-scrambled
experiment, before running it, TRACE the structured signal forward to the *first*
trainable map it reaches:

- **(a)** It reaches a *free* `Linear` under a *single* permutation, with an
  independent free projection at that site
  $\Rightarrow$ the control is **VACUOUS** (Theorem A: gap provably zero).
- **(b)** It enters as *fixed per-molecule data* $\times$ *observable* upstream of
  all `Linear` layers
  $\Rightarrow$ the control is **VALID** (Theorem B: gap meaningful).
- **(c)** The *same* projected vector is used under $\geq 2$ *inconsistent*
  permutations sharing a projection
  $\Rightarrow$ the control is **VALID (partial)** (Theorem C: gap meaningful,
  strength grows with number of inconsistent permutations).

Only (b) and (c) may be reported as evidence of inductive bias; (a) must be
discarded as uninterpretable.

---

## 6. Audit table for all 7 Levels

| Level | Structure | Path to trainable map | Condition | Status |
|-------|-----------|-----------------------|-----------|--------|
| L1 | no chemistry $\to$ operator routing; identity scramble | N/A | -- | sanity baseline |
| L2 | motif/cycle/spectral, each with own projection | single perm each | (a) | VACUOUS |
| L3 | motif reused under 2 perms for cycle phase | 2 inconsistent perms | (c) | PARTIAL |
| L4 | chem/dist, each with own projection | single perm each | (a) | VACUOUS |
| L5 | chem reused under 4 perms (RZ, RY, XX, YY) | 4 inconsistent perms | (c) | GENUINE |
| L6 | chem under 5 perms | 5 inconsistent perms | (c) | GENUINE |
| L7 | chem under 5 perms (U3 $\times 3$, CRX, CRY) | 5 inconsistent perms | (c) | GENUINE |
| Level G (L8) | $A(\mathrm{mol})$ gates IsingXX directly | fixed per-mol data $\times$ observable | (b) | GENUINE |

---

## 7. Scope and Limitations

The theorem concerns function-class **equality** in the worst case, i.e. over all
optimizers and unlimited data. Its claims are:

1. A **vacuous** control (Condition A) has provably zero *representational* gap. It
   does **not** follow that the structured model is bad or useless; only that the
   structured-vs-scrambled *gap* is uninterpretable as evidence of inductive bias,
   because both classes realize identical functions.

2. In practice, optimization may fail to fully exploit the absorbing
   reparametrization -- due to finite capacity, limited training time, initialization,
   or regularization that is not coordinate-symmetric. This can inflate the *apparent*
   gap for vacuous levels above the true zero. Such inflation is an artifact of the
   training procedure, not of representation, and must not be reported as bias.

3. Conditions B and C establish *inequality* of function classes, hence a genuine
   (nonzero) achievable gap. They do not by themselves quantify its magnitude; the
   size of the gap is an empirical question and may still be small (see project
   memory: bias is real but tiny at small $K$, and does not clearly scale with qubit
   count).

4. The permutation-absorption arguments assume the trainable domain is
   coordinate-symmetric (closed under permutation of input coordinates). If the
   trainable domain is itself structured/anisotropic (e.g. weight tying that is not
   permutation-invariant), Condition A may only partially hold and should be
   re-analyzed under Theorem C.

---

WRITTEN T8_absorbability.md
