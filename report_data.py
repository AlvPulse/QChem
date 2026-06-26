"""Single source of truth for the Results & Benchmarking chapter (docs/06).

Every number below is transcribed from a completed experiment artifact in this repo. The
`SOURCE` field on each block names the file the numbers came from, so the chapter and the
figures cite identical, traceable values. Nothing here is invented; figures import this module.

Provenance map
--------------
  _verify_absorb.py        -> ABSORB   (bit-exact absorbability proof, all 7 levels)
  _levelG_k4.log           -> DECOMP[K4] gate / levelG / meas_only (run_levelG_probe.py)
  _levelG_k6.log           -> DECOMP[K6] gate / levelG
  _levelG_k8.log           -> DECOMP[K8] gate
  _sweep.log               -> SWEEP    (run_bias_probe.py: gate-only replication + context)
  _alt_b_multiseed.py run  -> RANDSPLIT (15-seed random-split replication, K4)
  docs/04, docs/05         -> narrative cross-checks of the same numbers
"""

# ---------------------------------------------------------------------------
# 1. Absorbability proof: is run_benchmark.py's structured-vs-scrambled a valid control?
#    residual = max|structured - scrambled(single-permuted input)| with identical weights.
#    residual == 0  ->  a free upstream projection re-absorbs the scramble  ->  VACUOUS control.
#    SOURCE: _verify_absorb.py (deterministic, no training); docs/04 sec.1, docs/05 sec.2.3
# ---------------------------------------------------------------------------
ABSORB = {
    "source": "_verify_absorb.py",
    # level: (residual, status, reuse_structure)
    1: (None,    "no control",  "no chemistry->operator routing exists (scramble == structured)"),
    2: (0.0,     "VACUOUS",     "each input used under 1 perm via own free projection (bit-exact K=4,6,8)"),
    3: (0.18,    "partial",     "motif vector reused under 2 perms; cycle/spectral 1 each"),
    4: (0.0,     "VACUOUS",     "chem & distance each via own free projection, 1 perm each"),
    5: (1.3,     "genuine",     "one chem vector reused under 4 perms (RZ,RY,XX,YY)"),
    6: (1.6,     "genuine",     "one chem vector reused under 5 perms"),
    7: (1.4,     "genuine",     "one chem vector reused under 5 perms (U3x3,CRX,CRY)"),
}

# ---------------------------------------------------------------------------
# 2. Level-8 decomposition: structured (true adjacency) vs scrambled (random adjacency),
#    scaffold-grouped CV, pooled per-task paired Wilcoxon over the 12 Tox21 tasks.
#    config: gate=graph-gated entangler + single-qubit readout
#            levelG=graph-gated entangler + bond-correlator readout (Level 8)
#            meas_only=fixed ring entangler + bond-correlator readout (measurement isolated)
#    Fields: struct, scram (pooled mean ROC-AUC), median_dauc, npos/12, sign_p, wil_p,
#            run_deltas (per-seed run-level dAUC).
#    SOURCE: _levelG_k4.log, _levelG_k6.log, _levelG_k8.log (run_levelG_probe.py)
# ---------------------------------------------------------------------------
DECOMP = {
    ("gate", 4):      dict(struct=0.6457, scram=0.6404, median_dauc=+0.0044, npos=9, n=12,
                           sign_p=0.073,  wil_p=0.01709, run_deltas=[0.0049, 0.0057]),
    ("levelG", 4):    dict(struct=0.6415, scram=0.6326, median_dauc=+0.0078, npos=8, n=12,
                           sign_p=0.1938, wil_p=0.01709, run_deltas=[0.0163, 0.0016]),
    ("meas_only", 4): dict(struct=0.6072, scram=0.6337, median_dauc=-0.0271, npos=0, n=12,
                           sign_p=1.0,    wil_p=1.0,     run_deltas=[-0.0091, -0.0438]),
    ("gate", 6):      dict(struct=0.6409, scram=0.6379, median_dauc=+0.0026, npos=9, n=12,
                           sign_p=0.073,  wil_p=0.1331,  run_deltas=[0.012, -0.003, -0.0001]),
    ("levelG", 6):    dict(struct=0.6512, scram=0.6412, median_dauc=+0.0108, npos=9, n=12,
                           sign_p=0.073,  wil_p=0.0105,  run_deltas=[0.0165, 0.0069, 0.0066]),
    ("gate", 8):      dict(struct=0.6579, scram=0.6557, median_dauc=+0.0030, npos=7, n=12,
                           sign_p=0.3872, wil_p=0.1697,  run_deltas=[-0.0003, 0.0046]),
    # levelG K=8 deliberately absent: confirmation run deferred (docs/05 sec.6.2 note).
}

# ---------------------------------------------------------------------------
# 3. Context baselines per K (one-fold orientation values, first seed):
#    separable = entanglement removed; classical = capacity-unconstrained MLP on
#    [coarse feats || adjacency]; structured = gate-only pooled structured AUC.
#    SOURCE: _sweep.log "context (fold0)" lines (run_bias_probe.py)
# ---------------------------------------------------------------------------
CONTEXT = {
    4: dict(separable=0.6517, classical=0.7201, structured=0.6457),
    6: dict(separable=0.6513, classical=0.6817, structured=0.6391),
    8: dict(separable=0.6696, classical=0.7096, structured=0.6551),
}

# ---------------------------------------------------------------------------
# 4. Gate-only bias-vs-qubits sweep (independent replication of DECOMP gate row),
#    run_bias_probe.py qubit sweep; pooled per-task paired test.
#    SOURCE: _sweep.log "BIAS vs QUBITS" + per-K blocks
# ---------------------------------------------------------------------------
SWEEP = {
    4: dict(struct=0.6457, scram=0.6404, median_dauc=+0.0044, npos=9, n=12,
            sign_p=0.073,  wil_p=0.01709, run_deltas=[0.0049, 0.0057]),
    6: dict(struct=0.6391, scram=0.6371, median_dauc=+0.0020, npos=8, n=12,
            sign_p=0.1938, wil_p=0.1697,  run_deltas=[0.012, -0.003]),
    8: dict(struct=0.6551, scram=0.6554, median_dauc=+0.0015, npos=7, n=12,
            sign_p=0.3872, wil_p=0.5151,  run_deltas=[-0.0003]),
}

# ---------------------------------------------------------------------------
# 5. Random-split replication (15 seeds, K=4) -- second evaluation regime.
#    SOURCE: _alt_b_multiseed.py run, transcribed in docs/04 sec.3a / docs/05 sec.8.1
# ---------------------------------------------------------------------------
RANDSPLIT = dict(
    seeds=15, struct_gt_scram=13, mean_dauc=+0.0042, sign_p=0.0037, wil_p=0.0062,
    ci95=(+0.0013, +0.0071),
    ordering=dict(separable=0.664, scrambled=0.672, structured=0.673),
)

# ---------------------------------------------------------------------------
# 6. Dataset / protocol facts (for the experimental-setup tables).
#    SOURCE: docs/03, run_bias_probe.py, _sweep.log headers
# ---------------------------------------------------------------------------
PROTOCOL = dict(
    dataset="Tox21 block (12 nuclear-receptor / stress-response assays)",
    n_molecules=7823, n_scaffolds=2404, n_tasks=12,
    split="scaffold-grouped CV (Bemis-Murcko, GroupKFold); val carved scaffold-disjoint",
    adj_nnz_per_mol={4: 6.59, 6: 11.40, 8: 15.89},
    encoding="fixed Linear(5->2) -> RY,RZ per qubit (identical across variants)",
    optimizer="AdamW (q-params lr 1e-2, rest lr 1e-3), weight_decay 1e-4",
    primary_test="per-task paired Wilcoxon over pooled-CV ROC-AUC, 12 Tox21 tasks",
    n_layers=2,
)

# Pretty names used across tables/figures.
CONFIG_LABEL = {
    "gate": "Gate-gated (single-qubit readout)",
    "levelG": "Level 8 (gate + bond-correlator readout)",
    "meas_only": "Measurement-only (fixed ring entangler)",
}


def fmt_p(p):
    """Significance stars used in tables/figures."""
    if p != p:           # NaN
        return "n/a"
    if p < 0.01:
        return f"{p:.4g} **"
    if p < 0.05:
        return f"{p:.4g} *"
    return f"{p:.4g} (n.s.)"


if __name__ == "__main__":
    # Sanity dump so the module is self-checking.
    print("ABSORB levels:", {k: v[1] for k, v in ABSORB.items()})
    for key, d in DECOMP.items():
        print("DECOMP", key, "dAUC", d["median_dauc"], fmt_p(d["wil_p"]))
    print("RANDSPLIT", RANDSPLIT["mean_dauc"], fmt_p(RANDSPLIT["wil_p"]))
