import pennylane as qml
import torch
import torch.nn as nn
from src.models.gnn import MultiTaskGNN # Assuming your GNN is here

class StructuredQuantumFeatureMap(nn.Module):
    def __init__(self, in_features=64, n_qubits=5):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # We logically partition the 64-dim GNN output into 3 distinct functional blocks
        # This simulates the Spectral (S), Motif (M), and Diffusion (D) split
        self.proj_S = nn.Linear(in_features, 2) # Maps to Qubits 0, 1
        self.proj_M = nn.Linear(in_features, 2) # Maps to Qubits 2, 3
        self.proj_D = nn.Linear(in_features, 1) # Maps to Qubit 4

        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs_S, inputs_M, inputs_D):
            # 1. Superposition
            for q in range(self.n_qubits):
                qml.Hadamard(wires=q)

            # 2. Block S (Spectral-like) - R_y encoding
            qml.RY(inputs_S[0], wires=0)
            qml.RY(inputs_S[1], wires=1)
            qml.CNOT(wires=[0, 1]) # Intra-block entanglement

            # 3. Block M (Motif-like) - R_z encoding
            qml.RZ(inputs_M[0], wires=2)
            qml.RZ(inputs_M[1], wires=3)
            qml.CNOT(wires=[2, 3]) # Intra-block entanglement

            # 4. Block D (Diffusion-like) - R_x encoding
            qml.RX(inputs_D[0], wires=4)

            # 5. CROSS-BLOCK ENTANGLEMENT (The core advantage over classical concatenation)
            qml.CNOT(wires=[1, 2]) # Correlates S with M
            qml.CNOT(wires=[3, 4]) # Correlates M with D
            qml.CNOT(wires=[4, 0]) # Correlates D with S

            # 6. Observables
            observables = []
            # Local Bloch vectors (3 * 5 = 15 features)
            for q in range(self.n_qubits):
                observables.append(qml.expval(qml.PauliX(q)))
                observables.append(qml.expval(qml.PauliY(q)))
                observables.append(qml.expval(qml.PauliZ(q)))
                
            # Cross-block correlators (3 features)
            observables.append(qml.expval(qml.PauliZ(1) @ qml.PauliZ(2)))
            observables.append(qml.expval(qml.PauliZ(3) @ qml.PauliZ(4)))
            observables.append(qml.expval(qml.PauliZ(4) @ qml.PauliZ(0)))

            return observables

        self.qnode = qnode

    def forward(self, x):
        # x shape: [batch, in_features]
        # Squeeze values to [0, pi] for stable rotation angles
        s_angles = torch.sigmoid(self.proj_S(x)) * torch.pi
        m_angles = torch.sigmoid(self.proj_M(x)) * torch.pi
        d_angles = torch.sigmoid(self.proj_D(x)) * torch.pi

        batch_size = x.shape[0]
        out = []
        # Process batch. (vmap can be used in newer PennyLane versions, but loop is stable for custom QNodes)
        for i in range(batch_size):
            res = self.qnode(s_angles[i], m_angles[i], d_angles[i])
            out.append(torch.stack(res))

        return torch.stack(out) # Shape: [batch, 18]

class HybridStructuredQGNN(nn.Module):
    def __init__(self, num_tasks=12, hidden=64, n_qubits=5):
        super().__init__()
        self.gnn = MultiTaskGNN(num_tasks=num_tasks, hidden=hidden)
        
        # Override the classical head with our Structured Quantum Feature Map
        self.quantum_feature_map = StructuredQuantumFeatureMap(in_features=hidden, n_qubits=n_qubits)
        
        # 18 features come out of the QFM (15 local + 3 correlators)
        self.post_net = nn.Sequential(
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_tasks)
        )

    def forward(self, data):
        # 1. Classical GNN extraction (Modify according to your specific GNN forward logic)
        x = self.gnn.extract_node_features(data) # Pseudo-code: adapt to your GNN
        hg = self.gnn.readout(x, getattr(data, 'batch', None))
        
        # 2. Quantum Feature Mapping (No barren plateaus here, just fixed measurement logic)
        q_features = self.quantum_feature_map(hg).float()
        
        # 3. Post-processing to logits
        logits = self.post_net(q_features)
        return logits, q_features # Return q_features for diagnostics