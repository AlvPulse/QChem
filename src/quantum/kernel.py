import pennylane as qml
import torch
import torch.nn as nn

class StructuredQuantumKernel(nn.Module):
    def __init__(self, s_dim, m_dim, d_dim, n_qubits=5):
        super().__init__()
        self.n_qubits = n_qubits

        # Projections to 5D
        self.proj_S = nn.Linear(s_dim, n_qubits)
        self.proj_M = nn.Linear(m_dim, n_qubits)
        self.proj_D = nn.Linear(d_dim, n_qubits)

        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def qfm_circuit(x_S, x_M, x_D):
            # Block 1 (U_S): Angle encode x_S using Ry gates + CNOT ring
            for i in range(n_qubits):
                qml.RY(x_S[i], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i+1) % n_qubits])

            # Block 2 (U_M): Angle encode x_M using Rz gates + CNOT ring
            for i in range(n_qubits):
                qml.RZ(x_M[i], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i+1) % n_qubits])

            # Block 3 (U_D): Angle encode x_D using Ry gates + CNOT ring
            for i in range(n_qubits):
                qml.RY(x_D[i], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i+1) % n_qubits])

            # Local Paulis: <X_i>, <Y_i>, <Z_i>
            observables = []
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliX(i)))
                observables.append(qml.expval(qml.PauliY(i)))
                observables.append(qml.expval(qml.PauliZ(i)))

            # Correlators: <Z_i Z_{i+1}>
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ((i+1) % n_qubits)))

            return observables

        self.qnode = qfm_circuit

    def forward(self, x_S, x_M, x_D):
        """
        Computes the Quantum Feature Map (QFM) vector.
        Inputs:
            x_S: (B, s_dim) - Spectral features
            x_M: (B, m_dim) - Motif features
            x_D: (B, d_dim) - Diffusion features
        Output:
            qfm: (B, 20) - Quantum feature map vector
        """
        # Project to 5D
        p_S = self.proj_S(x_S)
        p_M = self.proj_M(x_M)
        p_D = self.proj_D(x_D)

        # Scale inputs (Optional: Apply activation like tanh or clamp)
        p_S = torch.tanh(p_S) * torch.pi
        p_M = torch.tanh(p_M) * torch.pi
        p_D = torch.tanh(p_D) * torch.pi

        batch_size = x_S.size(0)
        qfm_outputs = []
        for i in range(batch_size):
            res = self.qnode(p_S[i], p_M[i], p_D[i])
            qfm_outputs.append(torch.stack(res))

        return torch.stack(qfm_outputs)
