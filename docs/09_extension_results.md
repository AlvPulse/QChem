# Extension Results — robustness, mechanism, theory (publication-readiness)

*Consolidated, self-contained report of the experiments run to harden the Level-8 result toward a
high-stakes submission. Each is also woven into the main chapter (`docs/06`) at the cited location;
this file gathers the numbers, figures and takeaways in one place. Every value traces to a script and
a cached artifact (see §8). Companion: `docs/08` (gap audit). These extend — they do not replace — the
core result in `docs/06` (the absorbability theorem + the structured-vs-scrambled scaling bias).*

---

## 0. Summary

| # | Experiment | Question | Headline result | Fig | docs/06 |
|--:|-----------|----------|-----------------|-----|---------|
| 1 | K=8 confirmation | does the bias keep growing? | ΔAUC +0.0134, p=0.0024, **clears 7-way Holm** | fig01/04 | §II-B |
| 2 | Finite-shot | survive finite measurement? | shot-invariant to **32 shots/obs** (100% reals +) | fig15 | §B.7 |
| 3 | Device-noise | survive decoherence? | flat +0.0182→+0.0168 over p=0→20% | fig16 | §B.7 |
| 4 | Mechanism | is it really place-then-harvest? | correlation **5.1× on true bonds**; harvest 2.24× | fig14 | §C.4 |
| 5 | Representation probe | what do features encode? | structured encodes **topology** (λ_max R² 0.072 vs 0.040) | fig17 | §C.5 |
| 6 | Scaling lemma | *why* does measurement scale? | single-qubit readout provably blind to graph signal | — | §II-B.6 |
| 7 | Power analysis | what can 12 tasks detect? | min detectable ΔAUC **0.0066** @80% | — | §V.1 |

**One-line synthesis:** the measurement-encoded Level-8 bias **grows with K**, is **robust to finite
shots and device noise**, is **mechanistically a bond-local place-then-harvest** effect that **shows
up in the feature representation**, and has a **formal reason to scale** — i.e. the central claim is
now hardened on the axes a top venue (especially a quantum venue) probes.

---

## 1. K = 8 confirmation — the bias keeps growing
**Script:** `run_levelG_probe.py --qubits 8 --configs levelG` · **Source:** `_levelG_k8_final.log`.

| K | struct | scram | median ΔAUC | tasks + | Wilcoxon p | 7-way Holm-adj p |
|--:|-------:|------:|------------:|:-------:|:----------:|:----------------:|
| 4 | 0.6415 | 0.6326 | +0.0078 | 8/12 | 0.017 | 0.085 |
| 6 | 0.6512 | 0.6412 | +0.0108 | 9/12 | 0.011 | 0.063 |
| **8** | 0.6483 | 0.6328 | **+0.0134** | **10/12** | **0.0024** | **0.017 \*** |

**Takeaway.** Monotone growth in effect size *and* significance; K=8 is the **only cell that survives
a strict seven-way Holm correction** (α=0.05). Single seed (per-task paired test over 12 tasks retains
power; run-level +0.0155 reproduced in an independent run). The gate-only mechanism stays n.s. at K=8
(+0.0030, p=0.17), preserving the fade-vs-grow divergence.

---

## 2. Finite-shot robustness — survives real measurement
**Script:** `make_shots.py` · **Artifact:** `results/shots_K6.npz` · **Fig:** `fig15`.

Train Level-8 (K=6) exactly; at test time estimate every readout observable from N shots (each Pauli
measured independently — a conservative noise upper bound) and recompute the pooled bias over 12
realizations.

| Shots/observable | 32 | 128 | 512 | 4096 | exact |
|---|:--:|:--:|:--:|:--:|:--:|
| ΔAUC | +0.0184 | +0.0181 | +0.0180 | +0.0182 | +0.0182 |
| realizations positive | 100% | 100% | 100% | 100% | — |

**Takeaway.** The bias is **essentially shot-invariant down to 32 shots/observable** — it is *not* a
statevector artifact. Reason: the bias is a structured−scrambled *difference* (symmetric shot noise
cancels) over low-variance bond-pooled aggregates. This substantiates the hardware-native, O(K) claim.

---

## 3. Device-noise robustness — survives decoherence
**Script:** `make_noise.py` · **Artifact:** `results/noise_K6.npz` · **Fig:** `fig16`.

Pauli-twirled local depolarizing of per-qubit strength p (weight-w Pauli attenuated by `(1−p)^w`, so
the 2-local bond-correlators decay as `(1−p)²` — worst case for Level 8) + 2% readout bit-flip error.

| Depolarizing p | 0% | 1% | 2% | 5% | 10% | 20% |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| ΔAUC | +0.0182 | +0.0180 | +0.0180 | +0.0178 | +0.0175 | +0.0168 |

**Takeaway.** The bias is **flat across the noise sweep** — multiplicative decoherence rescales both
arms together, so the difference is preserved. Together with §2, the readout is robust to **both**
finite sampling and device noise. *(Analytic model — exact for global depolarizing; a gate-level
`default.mixed` simulation is the optional next step for maximal device realism.)*

---

## 4. Mechanism — direct place-then-harvest measurement
**Script:** `make_mechanism.py` · **Artifact:** `results/mechanism_K6.npz` · **Fig:** `fig14`.

On a trained Level-8 (K=6) circuit, measure the connected two-qubit correlator
`C_ij = ⟨Z_iZ_j⟩ − ⟨Z_i⟩⟨Z_j⟩` and split pairs into bonded (`A_ij>0`) vs non-bonded.

| Quantity | Value |
|---|---|
| `|C|` on bonded vs non-bonded pairs | 0.066 vs 0.013 (**5.1×**) — **PLACE** |
| on-bond correlation-mass fraction vs uniform | 0.71 vs 0.34 |
| harvested mass: true-A vs random-A pooling | 2.24× vs 0.98× (× uniform) — **HARVEST** |

**Takeaway.** The graph-gated entangler **places** quantum correlation on the molecule's true bonds;
bond-pooling with the true adjacency **harvests** it, while random-adjacency pooling does not. The
downstream AUC bias is the shadow of this measurable two-stage mechanism.

---

## 5. Representation probing — the bias is topology-aware
**Script:** `make_probe.py` · **Artifact:** `results/probe_K6.npz` · **Fig:** `fig17`.

Freeze the Level-8 (K=6) readout features; linear-probe (ridge, 5-fold CV R²) for molecular
properties, structured (true-A) vs scrambled (random-A) circuit.

| Probe target | structured R² | scrambled R² | Δ |
|---|:--:|:--:|:--:|
| λ_max(A) — largest adjacency eigenvalue *(topology)* | **+0.072** | +0.040 | **+0.032** |
| Fiedler value — algebraic connectivity *(topology)* | +0.141 | +0.134 | +0.007 |
| mean \|charge\| *(node control)* | −0.004 | −0.003 | ≈0 |
| aromatic fraction *(node control)* | +0.740 | +0.764 | −0.024 |

**Takeaway.** The structured circuit's *extra* representational content over scrambled is specifically
**topological** (λ_max ~80% more predictive), while node-feature encoding **ties** — the bias is a
topology-aware representation, not generic capacity. *Caveats:* absolute topology R² is low; the clean
signal is λ_max (Fiedler gap is within noise).

---

## 6. Scaling lemma — why the measurement readout scales (theory)
**Location:** `docs/06` §II-B.6 (Lemma 3).

The single-qubit readout `S: ρ ↦ (⟨X_i⟩,⟨Y_i⟩,⟨Z_i⟩)` is a function of the K one-qubit *marginals*
only and is therefore **blind** to the two-qubit correlations the graph-gated entangler writes. There
are `Θ(K²)` pairwise correlators but only `3K` single-qubit terms; for a sparse graph (`Θ(K)` bonds)
the signal lives in `Θ(K)` connected correlators that the bond-correlator readout `B_A` reads exactly
and `S` cannot. Hence gate-only (single-qubit) fades while Level-8 (bond-correlator) grows — and §4
measures the discarded signal (5.1× on-bond).

---

## 7. Power analysis
**Script:** `make_stats.py` (`min_detectable`) · **Location:** `docs/06` §V.1.

With 12 paired tasks and per-task SD ≈ 0.009, the **minimum detectable mean ΔAUC at 80% power**
(one-sided, α=0.05) is **≈ 0.0066** (z-approximation; Wilcoxon ARE ≈ 0.955). Level-8 effects
(+0.0078→+0.0134) clear it; gate-only effects (+0.0026→+0.0030) fall below — so "gate-only n.s." is
partly an honest power statement, consistent with Lemma 3.

---

## 8. Reproduce
```bash
python run_levelG_probe.py --qubits 8 --folds 3 --seeds 0 1 --configs levelG  # §1
python make_shots.py       # §2  -> fig15, results/shots_K6.npz
python make_noise.py       # §3  -> fig16, results/noise_K6.npz
python make_mechanism.py   # §4  -> fig14, results/mechanism_K6.npz
python make_probe.py       # §5  -> fig17, results/probe_K6.npz
python make_stats.py       # §7  -> results/stats_summary.json (Holm, effect sizes, power)
python make_schematics.py  # conceptual figs (Level-8 schematic, control-validity decision diagram)
```
Environment: `requirements.txt` (Python 3.12.10, CPU-only). The `results/*.npz` and `*.log` are
`.gitignore`d caches; regenerate from the commands above. All numbers also live in `report_data.py`.

---

## 9. What these results do and do not establish

**Establish (new this round).** The Level-8 bias (i) **grows** to K=8 and clears strict multiplicity;
(ii) is **robust** to finite shots and device noise; (iii) is **mechanistically** a bond-local
place-then-harvest effect that (iv) **shows up in the feature representation**; with (v) a **formal**
reason to scale and (vi) a **stated detectable-effect floor**.

**Do not establish.** None of this makes Level 8 beat a strong classical model, nor does it
generalize beyond Tox21 — **external validity (a second dataset) remains the open top-venue gate**
(see `docs/08`). The parameter-matched classical control is tracked separately.
