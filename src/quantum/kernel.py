import pennylane as qml
import torch
import torch.nn as nn

class StructuredQuantumKernel(nn.Module):
    def __init__(self, s_dim, m_dim, d_dim, n_qubits=5, use_projections=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.use_projections = use_projections

        # Projections to 5D
        if self.use_projections:
            self.proj_S = nn.Linear(s_dim, n_qubits)
            self.proj_M = nn.Linear(m_dim, n_qubits)
            self.proj_D = nn.Linear(d_dim, n_qubits)

        self.dev = qml.device("default.qubit", wires=n_qubits)

        torch.manual_seed(42)
        self.w1 = nn.Parameter(torch.randn(1, n_qubits, 3), requires_grad=False)
        self.w2 = nn.Parameter(torch.randn(1, n_qubits, 3), requires_grad=False)
        self.w3 = nn.Parameter(torch.randn(1, n_qubits, 3), requires_grad=False)

        @qml.qnode(self.dev, interface="torch")
        def qfm_circuit(x_S, x_M, x_D, eigenvecs):
            # Block 1: Spectral encoding
            for i in range(n_qubits):
                qml.RZ(x_S[i], wires=i)
            qml.StronglyEntanglingLayers(self.w1, wires=range(n_qubits))

            # Block 2: Spectral-modulated motif encoding
            modulated_angles = torch.mv(eigenvecs, x_M) * (torch.pi / 4.0)
            for i in range(n_qubits):
                qml.RX(modulated_angles[i], wires=i)
            qml.StronglyEntanglingLayers(self.w2, wires=range(n_qubits))

            # Block 3: Diffusion encoding
            for i in range(n_qubits):
                qml.RY(x_D[i], wires=i)
            qml.StronglyEntanglingLayers(self.w3, wires=range(n_qubits))

            # Measure full set of observables
            observables = []
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliX(i)))
                observables.append(qml.expval(qml.PauliY(i)))
                observables.append(qml.expval(qml.PauliZ(i)))
            for i in range(n_qubits):
                observables.append(qml.expval(
                    qml.PauliZ(i) @ qml.PauliZ((i+1) % n_qubits)
                ))
            return observables

        self.qnode = qfm_circuit

    def forward(self, x_S_vals, x_S_vecs, x_M, x_D):
        """
        Computes the Quantum Feature Map (QFM) vector using spectral-modulated encoding.
        Inputs:
            x_S_vals: (B, s_dim) - Spectral features (eigenvalues)
            x_S_vecs: (B, s_dim, s_dim) - Spectral features (eigenvectors)
            x_M: (B, m_dim) - Motif features
            x_D: (B, d_dim) - Diffusion features
        Output:
            qfm: (B, 20) - Quantum feature map vector
        """
        # Project to 5D
        if self.use_projections:
            p_S = self.proj_S(x_S_vals)
            p_M = self.proj_M(x_M)
            p_D = self.proj_D(x_D)
        else:
            p_S = x_S_vals
            p_M = x_M
            p_D = x_D

        # Phase scaling is crucial
        # Features are Standardized to mean 0, variance 1.
        # Scale to max absolute angle ~pi/2
        p_S = p_S * (torch.pi / 4.0)
        p_D = p_D * (torch.pi / 4.0)
        # Note: p_M is not scaled here because it is modulated inside the circuit.

        # We need the eigenvectors themselves to be properly scaled so they don't blow up the dot product
        # x_S_vecs has shape (B, 5, 5).
        # L2 normalize eigenvectors per graph just to be safe
        norms = torch.linalg.norm(x_S_vecs, dim=1, keepdim=True)
        # Avoid div by 0
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        p_vecs = x_S_vecs / norms

        batch_size = x_S_vals.size(0)
        qfm_outputs = []
        for i in range(batch_size):
            res = self.qnode(p_S[i], p_M[i], p_D[i], p_vecs[i])
            qfm_outputs.append(torch.stack(res))

        return torch.stack(qfm_outputs)
