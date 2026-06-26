# Ready-to-Paste Abstract & Novelty Statements

Drop-in text for the project report. Three lengths are provided; pick per venue. Numbers reflect
the completed K=4 and K=6 scaffold-CV results (K=8 Level-8 confirmation pending).

---

## A. Abstract (≈180 words)

Parameterized quantum circuits are often proposed as inductive-bias machines for molecular property
prediction: encode chemistry into the circuit's geometry and it should generalize better. Testing
this rigorously requires a control that holds capacity fixed while destroying only the
chemistry-to-circuit mapping. We show that the standard control — a fixed permutation of the
circuit's inputs — is **silently undone by the trainable layer that precedes the circuit**: we prove
bit-for-bit that, for several published encoding "levels," the permuted (scrambled) circuit and the
structured circuit are the *same function class*, so their comparison tests nothing. We then
introduce **Level 8**, a measurement-based design in which qubits are molecular-graph nodes and the
inductive bias is carried by *which observables are read*: two-qubit correlators pooled along the
real chemical bonds. Because the bond structure enters as fixed data multiplying a physical
observable, the control is provably non-absorbable. Under out-of-distribution scaffold
cross-validation, this yields a small but statistically significant topology bias that — unlike the
gate-only mechanism, which fades — **grows with qubit count** while remaining hardware-efficient
(O(K) local measurements).

---

## B. Novelty statement (≈90 words, for an "Contributions"/"Novelty" box)

**Contributions.** (1) We identify and prove a previously-unnoticed flaw in a common quantum
inductive-bias control: a fixed input permutation in front of a trainable projection is *absorbable*
— bit-exact identical to the structured circuit — so it isolates nothing. (2) We introduce **Level 8**,
a measurement-based bias in which two-qubit correlators are pooled along true molecular bonds, making
the control provably non-absorbable and the readout hardware-native and O(K)-scalable. (3) We show
the gate-encoded bias *fades* with circuit size whereas the measurement-encoded bias *grows*, the
first demonstration here of an inductive-bias mechanism that scales with qubit count.

---

## C. One-sentence "elevator" novelty

> We prove the field's standard structured-vs-scrambled control for quantum inductive bias is often
> mathematically vacuous (the trainable encoder absorbs it), and replace it with a measurement-based
> design — two-qubit correlators pooled along real molecular bonds — whose control is provably
> non-absorbable and whose advantage, uniquely, *grows* with qubit count.

---

## D. Key numbers (for tables/claims)

- Absorbability proof: scrambled ≡ structured **bit-exact (residual = 0.0)** at Levels 1/2/4
  (K = 4, 6, 8); genuine only at Levels 5–7 (residual 1.3–1.6).
- Topology bias (structured − scrambled), per-task paired Wilcoxon on pooled scaffold-CV predictions:
  - gate-only: K4 +0.0044 (p=0.017) → K6 +0.0026 (n.s.) → K8 +0.0030 (n.s.) — **fades**.
  - Level 8: K4 +0.0078 (p=0.017) → K6 **+0.0108 (p=0.011)**, higher absolute AUC — **grows**.
  - replication: 15-seed random split, 13/15 seeds, sign p=0.0037, Wilcoxon p=0.0062.
- Honesty caveat to include: a capacity-unconstrained classical MLP still leads by ~5–8 AUC points;
  this is an inductive-bias *existence-and-scaling* result, not a quantum-advantage claim.

---

## E. Suggested framing notes (not for pasting)

- Lead with the *negative* methodological result (absorbable control) — it is the strongest, most
  defensible, and most surprising contribution, and it motivates Level 8 naturally.
- Keep "inductive bias exists and scales" strictly separate from "quantum beats classical." The
  former is supported; the latter is not, and conflating them weakens the paper.
- The `meas_only` ablation (measurement readout *hurts* without graph-gated entanglement) is worth a
  sentence — it shows the mechanism is "place-then-harvest," not a measurement gimmick.
