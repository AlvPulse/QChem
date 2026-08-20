import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import argparse
from run_levelG_probe import GraphG, train_eval, CONFIGS
import pandas as pd
import json

from run_bias_probe import featurize, N_TASKS, FDIM, pairs_of, pos_weight, masked_bce, scaffold_folds

class GraphGEnhanced(GraphG):
    def __init__(self, k, entangler='graph', readout='graph', n_layers=2, out_dim=N_TASKS):
        super().__init__(k, entangler, readout, n_layers, out_dim)

        dev = qml.device('default.qubit', wires=k)
        PAIRS = self.pi.tolist()
        self.PAIRS = list(zip(self.pi.tolist(), self.pj.tolist()))

        @qml.qnode(dev, interface='torch')
        def circ(ry, rz, adj, theta, ringp, pairp, enc):
            for l in range(n_layers):
                for i in range(k):
                    qml.RY(enc[0] * ry[:, i], wires=i)
                    qml.RZ(enc[1] * rz[:, i], wires=i)

                for pidx, (i, j) in enumerate(self.PAIRS):
                    if entangler == 'graph':
                        # Enhanced: IsingXX + IsingZZ
                        qml.IsingXX(adj[:, i, j] * pairp[l, pidx], wires=[i, j])
                        qml.IsingZZ(adj[:, i, j] * pairp[l, pidx] * 0.5, wires=[i, j])
                    elif entangler == 'fixed' and j == (i + 1) % k:
                        qml.IsingXX(pairp[l, pidx], wires=[i, j])

                for i in range(k):
                    qml.RY(theta[l, i, 0], wires=i)
                    qml.RZ(theta[l, i, 1], wires=i)

                for i in range(k):
                    qml.CRZ(ringp[l, i], wires=[i, (i + 1) % k])

            obs = ([qml.expval(qml.PauliX(i)) for i in range(k)] +
                   [qml.expval(qml.PauliY(i)) for i in range(k)] +
                   [qml.expval(qml.PauliZ(i)) for i in range(k)])

            if readout == 'graph':
                obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in self.PAIRS]
                obs += [qml.expval(qml.PauliX(i) @ qml.PauliX(j)) for i, j in self.PAIRS]
                # Enhanced: add YY correlators
                obs += [qml.expval(qml.PauliY(i) @ qml.PauliY(j)) for i, j in self.PAIRS]

            return obs

        self.circ = circ

        # Update head in
        head_in = 3 * k + (3 * k if readout == 'graph' else 0)
        self.head = nn.Linear(head_in, out_dim)

    def forward(self, qf, adj):
        a = torch.atan(self.feat(qf))
        out = self.circ(a[:, :, 0], a[:, :, 1], adj, self.theta, self.ringp, self.pairp, self.enc)
        out = [o.float() for o in out]
        k, P = self.k, self.P
        feats = [torch.stack(out[:3 * k], -1)]
        if self.readout == 'graph':
            zz = torch.stack(out[3 * k:3 * k + P], -1)
            xx = torch.stack(out[3 * k + P:3 * k + 2 * P], -1)
            yy = torch.stack(out[3 * k + 2 * P:3 * k + 3 * P], -1)
            feats += [self._bond_pool(zz, adj), self._bond_pool(xx, adj), self._bond_pool(yy, adj)]
        return self.head(torch.cat(feats, -1))

def run_ablation(k=4, dataset='Tox21', max_mols=0):
    QF, AT, AR, Y, SCAF = featurize(k, [dataset])
    if max_mols > 0:
        QF, AT, AR, Y, SCAF = QF[:max_mols], AT[:max_mols], AR[:max_mols], Y[:max_mols], SCAF[:max_mols]
    splits = scaffold_folds(SCAF, n_splits=3)

    for model_name, ModelClass in [('GraphG_Base', GraphG), ('GraphG_Enhanced', GraphGEnhanced)]:
        print(f"Testing {model_name}...")
        for variant in ['structured', 'scrambled']:
            for fold, (tr, va, te) in enumerate(splits):
                model = ModelClass(k, entangler='graph', readout='graph')
                adj = AR if variant == 'scrambled' else AT
                QFt, At, Yt = torch.tensor(QF), torch.tensor(adj), torch.tensor(Y)

                B = 4
                out = model(QFt[:B], At[:B])
                print(f"{model_name} ({variant}) Forward Pass Success! Output shape: {out.shape}")
                break
            break

if __name__ == '__main__':
    run_ablation(k=4, dataset='Tox21', max_mols=32)
