# Quantum Circuit Levels: Algorithm Details

All seven levels live in `src/quantum_levels.py`. Each level defines four classes:
`Level{N}Classical`, `Level{N}Quantum` (the full model wrapper), and `Level{N}QuantumLayer` (the PennyLane circuit). A shared `SemanticFeatureExtractor` (GINEConv-based GNN) upstream of every level produces three 64-dimensional molecule-level vectors: **motif**, **cycle**, **spectral**, plus a **chemical** representation.

---

## Shared Upstream: SemanticFeatureExtractor (`src/features/semantic_extractor.py`)

Before any quantum level processes a molecule, the GNN encoder runs:

1. **Node embedding**: Five categorical atom features (atomic number, degree, formal charge, total H, is-aromatic) embedded and projected to `hidden_dim=64`.
2. **Edge encoding**: Three categorical bond features + one continuous 3D-distance feature combined into a `hidden_dim`-dimensional edge vector.
3. **Three GINEConv layers** with LayerNorm and ReLU residuals.
4. **Four attentional pooling heads**, each producing a 64-dimensional molecule vector:
   - **Motif** — local substructure aggregation
   - **Cycle** — augmented with aromatic-bond embeddings
   - **Spectral** — augmented with node-degree embeddings
   - **Chemical** — augmented with 8-dimensional continuous atom features

These four vectors are the inputs to every quantum level.

---

## Level 1 — Features → Models

**Theme**: Three parallel generic quantum circuits with attention aggregation.  
**File**: `quantum_levels.py`, lines 67–172.

### Architecture
```
motif_rep  (B,64) ──→ Linear(64, n_qubits) ──→ QuantumLayer_motif  ──→ (B, 3*n_qubits)
cycle_rep  (B,64) ──→ Linear(64, n_qubits) ──→ QuantumLayer_cycle  ──→ (B, 3*n_qubits)
spectral_rep(B,64)──→ Linear(64, n_qubits) ──→ QuantumLayer_spectral──→ (B, 3*n_qubits)
                       ↓ attention aggregation
                  task heads → (B, num_tasks)
```

### Quantum Circuit (generic)
```
for l in range(n_layers):
    RY(θ[l,i,0] + x[i]) for each qubit i        # generic encoding
    RZ(θ[l,i,1])         for each qubit i        # variational
    CRZ(ent[l,i], [i, (i+1)%n])  ring entangler  # entanglement
```
Measurements: `<X>`, `<Y>`, `<Z>` per qubit → 3·n_qubits features.

### Scrambled Control
Level 1's scrambled mode is structurally identical to structured because there is no chemistry→operator routing to destroy. It serves as a sanity-check baseline.

### Classical Equivalent
Three independent MLPs with hidden `inner_dim`, attention aggregation. `inner_dim` is auto-matched to equalise total parameter count.

### Role in Progression
Establishes the baseline: "does adding generic quantum circuits at all produce any improvement over a classical model?" If Level 1 shows no gap, the claim cannot rest on quantum capacity alone.

---

## Level 2 — Features → Operator Families

**Theme**: Motifs route to R_y, cycles to R_z, spectral to IsingXX.  
**File**: `quantum_levels.py`, lines 174–289.

### Chemistry → Operator Correspondence
| Feature Stream | Operator | Physical Rationale |
|---|---|---|
| Motif (local substructure) | `R_y(enc·x)` | Local real-valued rotations; motifs are real-space patterns |
| Cycle (aromaticity, rings) | `R_z(enc·x)` | Phase operator; aromatic systems accumulate phase |
| Spectral (global topology) | `IsingXX(enc·x)` | Interaction Hamiltonian; spectral modes reflect global coupling |

### Circuit (per layer, data re-uploading every layer)
```python
for l in range(n_layers):
    # ENCODING BLOCK — chemistry drives operator family
    for i in range(n_qubits):
        RY(enc_scale[0] * motif[perm_m[i]])    on qubit i
        RZ(enc_scale[1] * cycle[perm_c[i]])    on qubit i
    for i in range(n_qubits):
        IsingXX(enc_scale[2] * spectral[perm_s[i]], [i, (i+1)%n])

    # VARIATIONAL BLOCK — trainable, data-independent
    for i in range(n_qubits):
        RY(theta[l, i, 0])
        RZ(theta[l, i, 1])
    for i in range(n_qubits):
        CRZ(ent[l, i], [i, (i+1)%n])
```

### Scrambled Control
Three fixed random permutation arrays (`perm_m`, `perm_c`, `perm_s`) are drawn at model init and held constant. In scrambled mode, feature dimension `k` of a given stream routes to qubit `perm[k]` instead of qubit `k`, destroying the chemistry→qubit alignment while preserving the operator family assignment.

### Parameters
- `theta`: `(n_layers, n_qubits, 2)` — variational RY/RZ angles
- `ent`: `(n_layers, n_qubits)` — entanglement CRZ angles
- `enc_scale`: `(3,)` — learnable per-modality scaling, initialised to 1

### Key Insight
Level 2 is the first level where the structured vs. scrambled comparison is meaningful. The `enc_scale` parameters allow the model to down-weight a modality whose operator assignment is wrong, making the test conservative.

---

## Level 3 — Features → Operator Geometry

**Theme**: Modulation across feature modalities at the circuit level (quantum interference).  
**File**: `quantum_levels.py`, lines 291–439.

### What Changes from Level 2
Level 2 routes each feature stream to a separate operator family, but features remain independent. Level 3 introduces **cross-modality phase modulation**:

```
R_z( cycle[i]  +  α[i] · motif[i] )
```

Here `α` is a per-qubit learnable weight vector. Motif features modulate the cycle phase, creating a form of quantum interference unavailable in classical networks without explicit cross-product feature engineering.

### Circuit
```python
for l in range(n_layers):
    for i in range(n_qubits):
        RY(enc_scale[0] * motif[perm_m[i]])                         # motif → local
        RZ(enc_scale[1] * cycle[perm_c[i]]
           + alpha[i] * motif[perm_mm[i]])                          # cycle + modulation
    for i in range(n_qubits):
        IsingXX(enc_scale[2] * spectral[perm_s[i]], [i,(i+1)%n])   # spectral → entanglement
    # variational block (same as Level 2)
```

### Four Permutation Streams
Level 3 uses four independent permutations in scrambled mode:
- `perm_m` — motif→RY assignment
- `perm_c` — cycle→RZ assignment
- `perm_mm` — motif→RZ modulation assignment (separately scrambled)
- `perm_s` — spectral→IsingXX assignment

This means even the cross-modality coupling is destroyed in the scrambled control.

### Trainability Fixes Applied at Level 3
Level 3 is where several critical fixes were introduced after observing near-chance AUC in earlier designs:

1. **Data re-uploading every layer** — encoding applied at every depth layer, not just the input.
2. **Richer readout `<X>, <Y>, <Z>`** — replaced Z-only readout; exposes phase information.
3. **Learnable `enc_scale`** — per-modality scaling prevents one modality from dominating.
4. **Small variational init** — avoids near-identity circuit at the start of training.

### Classical Equivalent
FiLM-style feature modulation: `gamma * x + beta` where gamma and beta are linear functions of another feature stream. Implemented with LayerNorm and residual connections, matched in parameter count.

---

## Level 4 — 3D Spatial Entanglement

**Theme**: Bond distance modulates entanglement gate strength.  
**File**: `quantum_levels.py`, lines 443–559.

### New Input: 3D Distances
Level 4 introduces a second input stream alongside the GNN chemical representation:
- `chem` — chemical features (projected to n_qubits dimensions)
- `dist` — 3D Euclidean edge distances, pooled to molecule-level via `AttentionalAggregation`

### Circuit
```python
for l in range(n_layers):
    for i in range(n_qubits):
        RY(enc_scale[0] * chem[perm_c[i]])     on qubit i

    for i in range(n_qubits):
        # Distance-modulated entanglement: closer = stronger coupling
        CRZ(enc_scale[1] / (1.0 + dist[perm_d[i]]), [i, (i+1)%n])

    # variational block
```

The formula `enc / (1 + dist)` encodes a physically motivated inverse-distance coupling: atoms that are geometrically close (small dist) produce strong entanglement, while distant atoms are nearly decoupled. This mimics how quantum chemical interactions decay with distance.

### Scrambled Control
Two independent permutations:
- `perm_c` — chemistry→RY assignment
- `perm_d` — distance→CRZ assignment

### Classical Equivalent
`AttentionalAggregation` over edge embeddings, combined with chemistry features via an MLP. Edge embeddings include the 3D distance directly.

---

## Level 5 — Electronic Structure (Hückel Model)

**Theme**: Rotations encode electronegativity/partial charges; XX/YY encode bonding.  
**File**: `quantum_levels.py`, lines 561–640.

### Chemical→Operator Mapping
| Physical Quantity | Operator |
|---|---|
| Electronegativity / partial charge | `R_z` (phase) |
| Atomic orbital orientation | `R_y` (amplitude) |
| σ-bond / π-bond interaction | `IsingXX` |
| Cross-plane π interaction | `IsingYY` |

### Circuit
```python
for l in range(n_layers):
    for i in range(n_qubits):
        RZ(enc[0] * chem[perm_rz[i]])    # electronegativity
        RY(enc[1] * chem[perm_ry[i]])    # orbital orientation
    for i in range(n_qubits):
        IsingXX(enc[2] * chem[perm_xx[i]], [i, (i+1)%n])   # σ-bond
        IsingYY(enc[3] * chem[perm_yy[i]], [i, (i+1)%n])   # π-bond
    # variational block
```

Four permutation streams for scrambling: `perm_rz`, `perm_ry`, `perm_xx`, `perm_yy`.

### Classical Equivalent
Sequential `sigmoid → tanh` non-linearities on linearly projected features.

---

## Level 6 — 3D Electrostatic Mapping

**Theme**: Full single-qubit rotations + all-to-all CRZ modulated by pairwise feature products.  
**File**: `quantum_levels.py`, lines 642–721.

### Circuit
```python
for l in range(n_layers):
    for i in range(n_qubits):
        RX(enc[0] * chem[perm_rx[i]])    on qubit i
        RY(enc[1] * chem[perm_ry[i]])    on qubit i
        RZ(enc[2] * chem[perm_rz[i]])    on qubit i

    # All-to-all coupling modulated by pairwise feature products
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            CRZ(enc[3] * chem[perm_ci[i]] * chem[perm_cj[j]], [i, j])

    # variational block
```

The `chem_i · chem_j` product in the CRZ angle is a direct analogue of a pairwise electrostatic interaction: the coupling between two "atoms" (qubits) is proportional to the product of their chemical potentials. This achieves O(n_qubits²) interaction terms without a dense weight matrix.

Five permutation streams: `perm_rx`, `perm_ry`, `perm_rz`, `perm_ci`, `perm_cj`.

### Classical Equivalent
`PReLU` activations with cross-feature interaction terms (explicit outer products) matched in parameter count.

---

## Level 7 — Pharmacophore / Reactivity

**Theme**: U3 full single-qubit rotations per reactivity site; CRX/CRY pharmacophore dependencies.  
**File**: `quantum_levels.py`, lines 723–801.

### Chemical→Operator Mapping
`U3(θ, φ, λ)` is the most general single-qubit unitary. Routing three distinct molecular features (e.g., donor potential, acceptor potential, hydrophobicity) to the three angles gives the circuit maximum expressive power per qubit.

```python
for l in range(n_layers):
    for i in range(n_qubits):
        U3(enc[0]*chem[perm_u3a[i]],
           enc[1]*chem[perm_u3b[i]],
           enc[2]*chem[perm_u3c[i]],  qubit i)

    for i in range(n_qubits):
        CRX(enc[3] * chem[perm_crx[i]], [i, (i+1)%n])   # pharmacophore pair interaction X
        CRY(enc[4] * chem[perm_cry[i]], [i, (i+1)%n])   # pharmacophore pair interaction Y

    # variational block
```

Five permutation streams: `perm_u3a`, `perm_u3b`, `perm_u3c`, `perm_crx`, `perm_cry`.

### Classical Equivalent
`LeakyReLU` activations on linearly projected features, matched in parameter count.

---

## Measurement Strategy (All Levels)

All levels measure the full Bloch vector per qubit:
```
output = [<X>_0, <Y>_0, <Z>_0, <X>_1, ..., <Z>_{n-1}]   # shape: (B, 3·n_qubits)
```

This was a critical fix over the original Z-only readout (`[<Z>_0, ..., <Z>_{n-1}]`). Circuits whose encoding lives primarily in phase (R_z gates, IsingXX) produce near-zero `<Z>` expectation values at initialisation, giving no gradient signal. The `<X>` and `<Y>` components expose this phase information.

---

## Parameter Count Summary

For `n_qubits=4, n_layers=2`:

| Component | Parameters |
|---|---|
| `theta` (variational RY/RZ) | n_layers × n_qubits × 2 = 16 |
| `ent` (entanglement CRZ) | n_layers × n_qubits = 8 |
| `enc_scale` | 3–5 (level dependent) |
| `alpha` (Level 3 modulation) | n_qubits = 4 |
| Total quantum circuit | ~27–35 |
| Linear projections (classical part) | ~hundreds |

The classical baseline's `inner_dim` is solved to match the total quantum model parameter count.
