
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
            if ansatz == 'reupload':
                # Data Re-uploading: Interleave encoding and variational layers
                # Weights shape for reupload: (n_layers, n_qubits, 3)
                # (assuming simplified blocks or matching strong entangling per layer)
                # Actually, standard re-uploading puts encoding in every layer.
                for l in range(n_layers):
                    qml.AngleEmbedding(inputs, wires=range(n_qubits))
                    qml.StronglyEntanglingLayers(weights[l:l+1], wires=range(n_qubits))
            else:
                # Standard VQC
                qml.AngleEmbedding(inputs, wires=range(n_qubits))

                if ansatz == 'strong':
                    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                elif ansatz == 'mps':
                    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit

        if ansatz == 'strong':
            self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        elif ansatz == 'reupload':
            # Re-uploading repeats the block `n_layers` times.
            # Each block has 1 StronglyEntanglingLayer which takes shape (1, n_qubits, 3)
            # Total weights: (n_layers, 1, n_qubits, 3) -> Flattened to (n_layers, n_qubits, 3) for ease
            self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        elif ansatz == 'mps':
            self.weights = nn.Parameter(torch.randn(n_layers, n_qubits))

    def forward(self, x):
        if not torch.all(x >= -3.15) or not torch.all(x <= 3.15):
             x = torch.atan(x)

        results = self.qnode(x, self.weights)
        return torch.stack(results, dim=-1)
