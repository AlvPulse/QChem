
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
                elif ansatz == 'hea':
                    # Hardware Efficient Ansatz: RY, RZ per qubit, then CNOT ring
                    for l in range(n_layers):
                        for i in range(n_qubits):
                            qml.RY(weights[l, i, 0], wires=i)
                            qml.RZ(weights[l, i, 1], wires=i)
                        if n_qubits > 1:
                            for i in range(n_qubits):
                                qml.CNOT(wires=[i, (i+1)%n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit

        if ansatz == 'strong':
            self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        elif ansatz == 'reupload':
            self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        elif ansatz == 'mps':
            self.weights = nn.Parameter(torch.randn(n_layers, n_qubits))
        elif ansatz == 'hea':
            self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2))

    def forward(self, x):
        # Ensure inputs are within appropriate range for AngleEmbedding (if needed)
        # Using atan to map to [-pi/2, pi/2] roughly, or just scale
        if not torch.all(x >= -3.15) or not torch.all(x <= 3.15):
             x = torch.atan(x)

        results = self.qnode(x, self.weights)
        return torch.stack(results, dim=-1).float()

class QuantumEnsemble(nn.Module):
    def __init__(self, input_dim, n_estimators=4, n_qubits_per_est=4, n_layers=2, ansatz='strong', n_outputs=12, split_input=False):
        super().__init__()
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.n_qubits_per_est = n_qubits_per_est
        self.split_input = split_input
        self.n_outputs = n_outputs

        if split_input:
            if input_dim % n_estimators != 0:
                raise ValueError(f"Input dim {input_dim} must be divisible by n_estimators {n_estimators} when split_input=True")
            # In split mode, each estimator takes a chunk of input.
            # The chunk size must match n_qubits_per_est (since AngleEmbedding uses all qubits).
            if (input_dim // n_estimators) != n_qubits_per_est:
                 raise ValueError(f"Chunk size {input_dim // n_estimators} must match n_qubits_per_est {n_qubits_per_est}")
        else:
            # In copy mode, input is copied to each estimator.
            # Input dim must match n_qubits_per_est.
            if input_dim != n_qubits_per_est:
                raise ValueError(f"Input dim {input_dim} must match n_qubits_per_est {n_qubits_per_est} when split_input=False")

        self.estimators = nn.ModuleList([
            QuantumLayer(n_qubits_per_est, n_layers, ansatz) for _ in range(n_estimators)
        ])

        # Each estimator has its own head
        self.heads = nn.ModuleList([
            nn.Linear(n_qubits_per_est, n_outputs) for _ in range(n_estimators)
        ])

    def forward(self, x):
        # x: (B, input_dim)

        all_logits = []
        for i, (est, head) in enumerate(zip(self.estimators, self.heads)):
            if self.split_input:
                chunk_size = self.input_dim // self.n_estimators
                x_i = x[:, i*chunk_size : (i+1)*chunk_size]
            else:
                x_i = x

            # Quantum Layer
            q_out = est(x_i) # (B, n_qubits_per_est)

            # Head
            logits = head(q_out) # (B, n_outputs)
            all_logits.append(logits)

        # Stack: (B, n_estimators, n_outputs)
        stacked_logits = torch.stack(all_logits, dim=1)

        # Average logits (Ensemble averaging)
        avg_logits = torch.mean(stacked_logits, dim=1)

        return avg_logits, stacked_logits # Return stacked for diversity loss if needed later? Just avg for now.
