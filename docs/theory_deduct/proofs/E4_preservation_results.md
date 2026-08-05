# E4: Epsilon-Sufficiency / Toxicophore Preservation Results
*Tests T7 Thm 3.8 TL(G) locality assumption: toxicophores intact in single cluster?*
*Date: 2026-07-12 | Script: probe_info_preservation.py | N=500 molecules, Tox21+ToxCast*

---

## Results

### Preservation rates by toxicophore and K

```
Toxicophore         | K=4 rate | K=6 rate | K=8 rate | count
-----------------------------------------------------------------
aromatic_ring       |  0.479   |  0.264   |  0.096   |   167
nitro               |  1.000   |  1.000   |  0.800   |     5
amine_aro           |  0.595   |  0.691   |  0.691   |    42
carbonyl            |  0.981   |  0.940   |  0.857   |   413
amide               |  0.864   |  0.640   |  0.584   |   125
epoxide             |  0.818   |  0.591   |  0.364   |    22
hydroxyl_aro        |  0.649   |  0.532   |  0.493   |    77
-----------------------------------------------------------------
MEAN                |  0.769   |  0.665   |  0.555   |
```

*Note: halogen_aro pattern had SMARTS parse error ("cF,cCl,cBr") -- excluded. Use "[cF,cCl,cBr]" in future runs.*

### Summary

```
   K | mean_preservation | TL(G) >0.7?
   4 |             0.769 | MARGINAL
   6 |             0.665 | FAIL
   8 |             0.555 | FAIL
```

---

## T7 Thm 3.8 TL(G) Assessment

**RESULT: TL(G) assumption is VIOLATED at K=6 and K=8 for most toxicophores.**

### Critical finding: aromatic ring splitting

Aromatic ring preservation: 47.9% (K=4), 26.4% (K=6), 9.6% (K=8).

At K=8, **90% of aromatic rings are SPLIT across multiple clusters.** This severely violates
the TL(G) assumption. Benzene has 6 atoms; at K=8 with ~20-30 heavy atoms per drug-like
molecule, this means the 8 clusters have ~3 atoms each -- too small to contain a 6-atom ring.

This is a GEOMETRIC necessity: when K >= n_atoms / max_toxicophore_size, splitting is unavoidable.
For aromatic rings (6 atoms) and K >= n/6, the ring will often be split. At modal n~25 and K=8:
mean cluster size = 25/8 = 3.1 atoms/cluster -- smaller than the 6-atom ring.

### Small toxicophores are preserved

Nitro (3 atoms): 100% at K=4,6; 80% at K=8 -- fits in one cluster.
Carbonyl (2 atoms): 98/94/86% -- very small, rarely split.
Amide (4 atoms): 86/64/58% -- medium, splitting increases with K.

### What this means for the model

**The model STILL WORKS despite TL(G) violation.** At K=8, AUC ~= 0.70 and dAUC > 0
(from stats_summary.json). This means Thm 3.8's TL(G) assumption is SUFFICIENT but NOT NECESSARY.

**Why the model works despite ring splitting:**
1. Bond correlators cross cluster boundaries: the bond adjacency A_coarse encodes inter-cluster
   bond weights. If ring atoms span clusters i and j, A_coarse[i,j] > 0 (aromatic bond weight),
   and the circuit entangles qubits i and j via IsingXX(A[i,j]*theta). The BOND CORRELATOR
   <Z_iZ_j> captures the inter-cluster aromatic bond even though the ring is split.
2. Single-qubit aromaticity feature: the coarse node feature qf includes aromaticity fraction
   (mean of in_ring/is_aromatic for atoms in the cluster). Even split clusters encode
   partial aromaticity signal.
3. The TC-QIC bottleneck works on a DIFFERENT level: the relevant information is the
   BOND CONNECTIVITY PATTERN (which bonds are aromatic), not the INTACT RING (which requires
   all atoms in one cluster). The coarse graph A and the bond correlators B_A[i] = sum_j A_ij C_ij
   encode connectivity even after ring splitting.

### Formal implication: revision of Thm 3.8

T7 Thm 3.8 should be labeled more carefully:

REVISED SCOPE: Thm 3.8 (conditional macro-topology sufficiency) holds for SMALL TOXICOPHORES
(nitro, carbonyl, amide) at all tested K, where TL(G) is approximately satisfied.
For LARGE TOXICOPHORES (aromatic rings, PAHs, macrocycles), TL(G) fails at K >= 6,
and the theorem does NOT apply as stated.

The empirical observation that the model still works (dAUC > 0 at K=8) suggests an EXTENDED
MECHANISM: the bond correlator B_A captures inter-cluster ring connectivity, providing an
approximate sufficiency result even when TL(G) fails. This extended mechanism is NOT
currently proven in T7 and represents a theoretical gap.

### epsilon_spectral quantification

The information loss epsilon_spectral = I(G;Y) - I(C(G);Y) can be bounded by the non-preservation:
For aromatic-ring-relevant tasks (NR-ER, NR-AhR, SR-ARE likely involving aromaticity):
  epsilon_spectral ~= loss from ring splitting = (1 - preservation_rate) * I(ring_indicator; Y)
At K=8: (1 - 0.096) = 90.4% of aromatic ring information is lost to cluster splitting.
This is a LARGE loss for aromaticity-sensitive tasks, explaining the AUC ceiling (0.61-0.66)
at K=8 for those tasks.

For small-toxicophore tasks (carbonyl-based toxicity): epsilon_spectral ~= (1 - 0.857) = 14.3%.

---

## Connection to E3

E3 showed 93% low-frequency energy in coarse features (spectral preservation).
E4 shows 56% mean toxicophore preservation rate at K=8.

These are DIFFERENT measurements:
- E3: energy in the smooth low-frequency LAPLACIAN modes (spectral preservation) -- HIGH
- E4: structural preservation of SPECIFIC PATTERNS (toxicophore-level) -- LOW for rings

The discrepancy shows: the coarse features are spectrally smooth (E3) but structurally
fragmented at the toxicophore level (E4). The Laplacian low-pass filters SMOOTH VARIATION;
it does NOT preserve arbitrary local substructure topology.

This is the key tension in TC-QIC: the spectral bottleneck is topology-preserving in the
SMOOTH-VARIATION sense, not in the PATTERN-MATCHING sense.

---

## Status: E4 DONE -- TL(G) VIOLATED for aromatic rings at K=6,8; small toxicophores OK

Key implications:
1. T7 Thm 3.8 scope must be tightened to small toxicophores OR extend theory to handle ring splitting
2. The model's success despite ring splitting comes from inter-cluster bond correlators (TC-QIC extended mechanism)
3. AUC ceiling on aromaticity tasks (~0.61-0.66) is PARTIALLY explained by 90% ring splitting at K=8
4. Future work: prove an extended Thm 3.8 that handles ring splitting via bond correlators crossing clusters

Next: E5 K=10 results (check if dAUC still grows despite increased ring splitting at K=10).
