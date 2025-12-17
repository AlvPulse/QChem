
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pennylane import numpy as np

# A hybrid model: GNN Encoder -> Quantum Layer -> Output
# This allows handling high complexity features via GNN and using QML for the final classification/correlation.

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, input_dim, output_dim):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Projection from input feature space to n_qubits (for angle embedding)
        self.pre_net = nn.Linear(input_dim, n_qubits)

        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Define weights as PyTorch parameters
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

    def forward(self, x):
        # x: [batch, input_dim]
        batch_size = x.shape[0]
        # Map input features to rotation angles
        angles = torch.tanh(self.pre_net(x)) * np.pi

        # Define the QNode locally or as a method to capture 'self' if needed,
        # but better to define it outside and pass parameters.

        # We need to broadcast execution over the batch.
        # PennyLane supports parameter broadcasting.

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # inputs shape: [batch_size, n_qubits]
            # weights shape: [n_layers, n_qubits, 3]

            # Encoding
            # We broadcast operations.
            for q in range(self.n_qubits):
                qml.Hadamard(wires=q)
                # inputs[:, q] is the q-th feature across batch
                qml.RZ(inputs[:, q], wires=q)
                qml.RY(inputs[:, q], wires=q)

            # Variational layers
            # StronglyEntanglingLayers doesn't support batch broadcasting for 'weights' typically combined with 'inputs' broadcasting in older versions?
            # But let's check recent versions.
            # If not, we iterate. But iteration is slow.
            # Using BasicEntanglerLayers or manually writing layers might be safer for broadcasting.

            # For simplicity and speed in this context, let's assume broadcasting works
            # or simply use a manual ansatz that broadcasts.

            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)

                # Entangling
                for q in range(self.n_qubits):
                    qml.CNOT(wires=[q, (q+1)%self.n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Execute circuit
        # We pass 'angles' as inputs.
        # If broadcasting is supported, this returns a list of tensors, each shape [batch_size].
        results = circuit(angles, self.weights)

        # Stack results: List of [batch_size] -> [batch_size, n_qubits]
        return torch.stack(results, dim=-1)

# Hybrid Model
from src.models.gnn import MultiTaskGNN

class HybridQGNN(nn.Module):
    def __init__(self, num_tasks=12, hidden=64, n_qubits=4, n_layers=2):
        super().__init__()
        # Use the classical GNN as a powerful encoder
        self.gnn = MultiTaskGNN(num_tasks=num_tasks, hidden=hidden)
        # Remove the final head of the GNN to replace with Quantum Head
        # The GNN head in src/models/gnn.py outputs 'num_tasks'
        # We want the intermediate representation 'hg' (hidden graph rep)
        # But MultiTaskGNN forward returns logits.
        # We can inherit and override forward, or modify MultiTaskGNN.
        # Let's override forward here by monkey-patching or just copying the structure.
        # Easier: Modify src/models/gnn.py to optionally return embedding.
        # But for now I'll just copy the encoder part.

        self.encoder_proj = self.gnn.proj
        self.encoder_g1 = self.gnn.g1
        self.encoder_g2 = self.gnn.g2
        self.encoder_g3 = self.gnn.g3
        self.encoder_bn1 = self.gnn.bn1
        self.encoder_bn2 = self.gnn.bn2
        self.encoder_node_embs = self.gnn.node_embs
        self.readout = self.gnn.readout

        # Quantum Layer
        # Input to Quantum Layer will be 'hidden' size from GNN
        # But 'hidden' is 128 usually. 4 qubits is too small to encode 128 dims directly without compression.
        # The QuantumLayer class has a pre_net that projects input_dim -> n_qubits.
        self.quantum_layer = QuantumLayer(n_qubits, n_layers, input_dim=hidden, output_dim=num_tasks)

        # We might need a post-processing layer to match range or do independent tasks
        # But QuantumLayer output_dim=num_tasks, so it returns [batch, num_tasks]
        # Range of PauliZ is [-1, 1]. Sigmoid expects logits.
        # So we might want a scaling layer or just treat these as logits (though bounded).
        # Better: Quantum Layer outputs features -> Classical Linear -> Logits.
        # Or: Quantum Layer outputs num_tasks values.
        # Let's try Quantum Layer -> Linear(n_qubits, num_tasks) to allow scaling/bias.
        # We changed circuit to output n_qubits measurements.
        self.post_net = nn.Linear(n_qubits, num_tasks)

    def forward(self, data):
        # Encoder (Classical GNN)
        x_cat = []
        Z, deg, chg, Hs, aro = data.x[:,0], data.x[:,1], data.x[:,2], data.x[:,3], data.x[:,4]
        for emb, feat in zip(self.encoder_node_embs, [Z,deg,chg,Hs,aro]):
            x_cat.append(emb(feat.clamp(min=0, max=emb.num_embeddings-1)))
        x = torch.cat(x_cat, dim=-1)
        x = self.encoder_proj(x)

        x = F.relu(self.encoder_g1(x, data.edge_index, data.edge_attr))
        x = self.encoder_bn1(x)
        x = F.relu(self.encoder_g2(x, data.edge_index, data.edge_attr))
        x = self.encoder_bn2(x)
        x = F.relu(self.encoder_g3(x, data.edge_index, data.edge_attr))

        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        hg = self.readout(x, batch)

        # Quantum Head
        q_out = self.quantum_layer(hg)

        # Ensure q_out is float32 (PennyLane often returns float64)
        q_out = q_out.float()

        # Post-processing to logits
        logits = self.post_net(q_out)

        return logits
