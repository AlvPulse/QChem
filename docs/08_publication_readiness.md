# Publication-Readiness Audit — what's missing for a high-stakes journal

*A systematic gap analysis of the current project state (as of the Results chapter `docs/06` +
methods `docs/07`) against the bar for a top venue (Nature Machine Intelligence / PRX Quantum /
NeurIPS-Datasets&Benchmarks). Each row says what we **have**, the **gap**, its **severity**, the
**effort** to close, and **where in the repo** it lives. Severity: 🔴 blocker · 🟠 major · 🟡 minor.
Effort: L = hours · M = ~a day · H = days+.*

---

## A. One-paragraph verdict

The project has a **genuinely publishable core** — the *absorbable-control* finding (Proposition 1)
is a novel, broadly-applicable methodological result, and the *measurement-encoded, non-absorbable,
scaling* inductive bias (Level 8) is a clean positive result with honest statistics. That core is
currently at the level of a **strong methods note / workshop paper**. To clear a **high-stakes
journal** it was missing three things, in priority order: **(1) external validity** (a single 12-task
dataset is disqualifying on its own — *still open*), **(2) a parameter-matched classical control on
the coarse graph** (without it the "inductive bias, not capacity" claim has a hole — *in progress
separately*), and **(3) a finite-shot / noise analysis** to substantiate the "hardware-native, O(K),
scalable" selling point (*✅ done — §B.7: finite-shot `make_shots.py` + device-noise `make_noise.py`*).
Closing (1) is now the single remaining gate to a top venue; (3) is fully addressed, and the theory/
power/artifact additions (Lemma 3, computed power, `requirements.txt`) are already in.

---

## B. Gap matrix

### B.1 Experimental completeness
| Requirement | Have | Gap | Sev | Effort | Repo |
|---|---|---|:--:|:--:|---|
| Multiple datasets / task families | Tox21 (12 tasks); merged Tox21+ToxCast exists but unused by the probe | **No second endpoint family** — the scaling pattern is shown on one dataset | 🔴 | M–H | `src/data_loader.py` (merged loader exists) |
| Parameter-matched classical control | classical MLP is **unconstrained (~10× params)** | a classical model on the *same coarse graph + adjacency* with *matched capacity* is untested — the obvious "could classical do it too?" control | 🔴 | L | `run_bias_probe.py:177` `ClassicalRef` (just constrain it) |
| Statistical power (seeds) | K4: 2 seeds, K6: 3, **K8: 1** | headline K=8 cell is single-seed; need ≥5 seeds/cell with CIs on every cell | 🔴 | M | `run_levelG_probe.py --seeds` |
| Strong absolute-performance baselines | classical MLP on coarse feats | no full-graph **GINE GNN** / **RF-on-fingerprints** number to anchor absolute AUC | 🟠 | L–M | `src/models/`, `run_classical.py` exist |
| Other quantum baselines | structured/scrambled/separable/meas_only | no comparison to **quantum kernels** or alternative ansatzes | 🟡 | M | `src/benchmark_sqk.py` exists |
| Sensitivity analyses | fixed K-clustering, fixed hyperparams | no sweep over **coarse-graining K choice, clustering method, lr, layers** | 🟡 | M | — |

### B.2 The scalability / hardware claim (central selling point)
| Requirement | Have | Gap | Sev | Effort | Repo |
|---|---|---|:--:|:--:|---|
| Finite-shot analysis | ✅ **DONE** — `make_shots.py`/fig15: bias is shot-invariant down to **32 shots/observable** (ΔAUC +0.0184, 100% realizations +); backs the hardware-native O(K) claim | (was 🔴/🟠) | — | `make_shots.py`, `docs/06` §B.7 |
| Noise model | none | no **depolarizing / readout-error** robustness; a quantum venue expects at least a NISQ noise sweep | 🟠 | M | PennyLane `default.mixed` |
| Real-hardware demonstration | none | optional but high-impact for a quantum venue (small K on IBM/IonQ) | 🟡 | H | — |
| Scaling beyond K=8 | K∈{4,6,8} (statevector-bound) | the O(K) claim is asymptotic; argued not measured beyond 8 (acceptable IF the shot analysis backs it) | 🟠 | — | tie to shot analysis |

### B.3 Theory
| Requirement | Have | Gap | Sev | Effort | Repo |
|---|---|---|:--:|:--:|---|
| Control-validity theorem | **Proposition 1 + Corollary 2** (absorbability) | — solid | ✅ | — | `docs/06` §I.2.1 |
| *Why measurement scales* | hand-wavy ("single-qubit readout discards O(K²) correlations") | no **formal lemma**: single-qubit readout is O(K) functions of the state; the graph-structured signal lives in the O(#bonds) two-qubit correlators it cannot see | 🟠 | M | new §; mechanism data in `make_mechanism.py` supports it |
| Generalization argument | capacity-matched ⇒ gap is bias | could connect to a **sample-complexity / margin** statement (ties to the learning-curve experiment) | 🟡 | M | LC experiment in progress |

### B.4 Interpretability / mechanism (reviewer Q11)
| Requirement | Have | Gap | Sev | Effort | Repo |
|---|---|---|:--:|:--:|---|
| Mechanism evidence | **place-then-harvest correlation localization** (5.1× on bonds) | strong, but it's one view | ✅/🟠 | — | `make_mechanism.py`, fig14 |
| Representation probing | none | no **t-SNE/UMAP** of quantum features, no **probing classifier** for chemical properties, no per-task "what chemistry does the bias help?" | 🟡 | M | cached features → cheap |

### B.5 Statistics & methodology
| Requirement | Have | Gap | Sev | Effort | Repo |
|---|---|---|:--:|:--:|---|
| Paired tests, multiplicity, effect sizes | Wilcoxon, sign, **Holm**, bootstrap CI, rank-biserial, two regimes | — strong | ✅ | — | `make_stats.py`, Table III.2 |
| CIs on **every** cell | only reproduction cells + random-split | need bootstrap CIs per published cell (needs the multi-seed per-task arrays) | 🟠 | L | tie to seeds |
| Power analysis | *stated* ("draws power from pairing") | not **computed** (min detectable ΔAUC at 80% power) | 🟡 | L | — |
| Pre-registration framing | hypotheses stated before results in §0 | could add an explicit **pre-registered prediction** box | 🟡 | L | `docs/06` §0 |

### B.6 Scholarship & artifacts
| Requirement | Have | Gap | Sev | Effort | Repo |
|---|---|---|:--:|:--:|---|
| Related work | a **sketch** with "verify bibliographic details" | needs a real, verified literature review (10–25 refs), not author-year placeholders | 🟠 | M | `docs/06` §0.1 |
| Code release | scripts + `report_data.py` single-source + seeds + versions | no **`requirements.txt`/conda lock**, no data-availability statement, no Zenodo/DOI archive | 🟠 | L | repo root |
| Broader-impact / ethics | none | some venues (NMI) require an impact statement | 🟡 | L | — |
| Compute/energy reporting | asymptotics argued | no wall-clock/FLOP/energy table | 🟡 | L | — |

---

## C. The three blockers, explained

1. **External validity (🔴).** Every high-stakes reviewer asks "does this hold beyond one dataset?"
   A measurement-based bias that grows with K on *Tox21 only* is a single data point about the
   *mechanism*. Repeating the K4→K6→K8 gate-fades / Level-8-grows pattern on **one more task family**
   (a ToxCast sub-panel, BBBP, BACE, or a regression target like ESOL/FreeSolv) converts it from "a
   result on Tox21" to "a property of the mechanism." The merged loader already exists; the main cost
   is re-running the coarse-graph featurization + the probe for a second label block.

2. **Parameter-matched classical control (🔴, but cheap).** The current classical baseline is
   *unconstrained* (~10× params), so it answers "is quantum better?" (no) but **not** the question
   that matters: *given the same coarse graph and adjacency, does a capacity-matched classical model
   capture the same topology signal?* If a matched classical GNN/MLP on `[coarse feats ‖ A]` also
   beats its scrambled-A version by ~the same ΔAUC, the quantum mechanism is not special; if it does
   **not**, that is a *much stronger* positive result. This is a few hours of work (`ClassicalRef`
   already exists — just match its width to the quantum param count and run structured-vs-scrambled
   on it) and it is the single highest-value-per-hour addition in this document.

3. **Finite-shot / noise analysis (🔴–🟠) — ✅ shot part DONE.** The distinctive claim is
   "hardware-native, O(K), scalable measurement readout." The finite-shot simulation
   (`make_shots.py`, docs/06 §B.7) now backs it: the structured−scrambled bias is **shot-invariant
   down to 32 shots per observable** (ΔAUC +0.0184, 100% of realizations positive), because the bias
   is a difference (symmetric noise cancels) over low-variance pooled correlators. The **device-noise
   sweep is also done** (`make_noise.py`, fig16): Pauli-twirled depolarizing with the 2-local readout
   decaying as `(1−p)²` plus 2% readout error leaves the bias flat (+0.0182→+0.0168) across
   `p = 0→20%`. A gate-level `default.mixed` simulation is the optional next step for maximal realism.

---

## D. Critical path to submission (minimal viable additions, ordered)

| # | Action | Closes | Sev | Effort |
|--:|--------|--------|:--:|:--:|
| 1 | Parameter-match the coarse-graph classical control; run structured-vs-scrambled-A on it | core-claim hole | 🔴 | L |
| 2 | Re-run Level 8 at K∈{4,6,8} with **5 seeds**; bootstrap CI per cell | power, K=8 n=1 | 🔴 | M |
| 3 | ✅ DONE — finite-shot (`make_shots.py`) + device-noise (`make_noise.py`) | scalability claim | ✅ | — |
| 4 | Second dataset/task family; replicate the fade-vs-grow pattern | external validity | 🔴 | M–H |
| 5 | Learning-curve / sample-efficiency experiment (already in progress) | "is it really a bias?" | 🟠 | L–M |
| 6 | Formal lemma: why bond-correlator readout preserves graph signal single-qubit loses | theory depth | 🟠 | M |
| 7 | Strong baselines (full-graph GINE, RF-fingerprints) for absolute context | baseline credibility | 🟠 | L–M |
| 8 | Verified literature review (replace the §0.1 sketch) + `requirements` lock + data-availability | scholarship/artifacts | 🟠 | M |
| 9 | Representation probing (t-SNE + probing classifier) | interpretability (Q11) | 🟡 | M |

Items 1–4 are the gate to a high-stakes venue; 5–9 are what move it from "accept with major revisions"
toward "accept." Items 1, 5 are partly done or trivial and would land quickly.

---

## E. Venue-fit note (the gaps differ by target)

- **PRX Quantum / Quantum** — the most natural home for "a correctly-controlled, scalable quantum
  inductive bias + the absorbability theorem." Here the **shot/noise analysis (item 3)** and the
  **formal scaling lemma (item 6)** are the load-bearing additions; a second dataset matters less than
  the quantum-rigor items.
- **Nature Machine Intelligence / NeurIPS** — will weight **external validity (item 4)**, **strong
  baselines (item 7)**, and **broader significance** most; the absorbable-control finding is the hook,
  but they will want the empirical breadth.
- **A "negative-results / methodology" framing** — the absorbable-control result *alone* (Proposition
  1 + the bit-exact proof + the decision procedure) is publishable as a methods caution with much less
  added work; this is the lowest-risk path if time is the binding constraint.

**Honest strategic read:** the work will *not* become a "quantum beats classical" paper, and it should
not try to be. Its publishable identity is **methodological** (how to test a quantum inductive bias
without fooling yourself) **+ a clean scalable positive mechanism**. Picking the venue first
determines which of items 1–9 are mandatory versus optional.
