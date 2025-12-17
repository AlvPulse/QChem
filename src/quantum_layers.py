
import pennylane as qml
import torch
import torch.nn as nn

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers=2, ansatz='strong'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz_type = ansatz
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))

            if ansatz == 'strong':
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            elif ansatz == 'mps':
                # Use BasicEntanglerLayers as a robust alternative representing local connectivity
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit

        if ansatz == 'strong':
            # Shape: (n_layers, n_qubits, 3)
            self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        elif ansatz == 'mps':
            # BasicEntanglerLayers uses 1 param per wire per layer (n_layers, n_wires)
            self.weights = nn.Parameter(torch.randn(n_layers, n_qubits))

    def forward(self, x):
        # Scale if not already scaled (hybrid model does atan for fft, linear layer handles itself)
        # Hybrid model's Linear output is unbounded, so atan acts as a bound.
        if not torch.all(x >= -3.15) or not torch.all(x <= 3.15):
             x = torch.atan(x)

        results = self.qnode(x, self.weights)
        return torch.stack(results, dim=-1)
