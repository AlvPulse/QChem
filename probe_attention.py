import torch
import torch.nn as nn
import numpy as np
import argparse
from run_levelG_probe import GraphG
from run_bias_probe import featurize

class GraphGInterpret(GraphG):
    def __init__(self, k, entangler='graph', readout='graph', n_layers=2, out_dim=12):
        super().__init__(k, entangler, readout, n_layers, out_dim)

    def forward_with_interpretability(self, qf, adj):
        a = torch.atan(self.feat(qf))
        out = self.circ(a[:, :, 0], a[:, :, 1], adj, self.theta, self.ringp, self.pairp, self.enc)
        out = [o.float() for o in out]
        k, P = self.k, self.P
        feats = [torch.stack(out[:3 * k], -1)]

        # We want to extract the exact correlation vectors before pooling
        if self.readout == 'graph':
            zz = torch.stack(out[3 * k:3 * k + P], -1)        # (B,P) -> Raw Z_i Z_j correlations
            xx = torch.stack(out[3 * k + P:3 * k + 2 * P], -1) # (B,P) -> Raw X_i X_j correlations

            # Unweighted pooling
            feats += [self._bond_pool(zz, adj), self._bond_pool(xx, adj)]

            # Compute interpretability map (which edges drove the prediction?)
            # b_i = sum_j A[i,j] <Zi Zj>
            # The exact edge attribution for the Z correlator on edge (i,j) for node i is A[i,j] * <Zi Zj>

            # W[i,j] = A[i,j] * corr
            zz_edge_attrib = adj[:, self.pi, self.pj] * zz
            xx_edge_attrib = adj[:, self.pi, self.pj] * xx

            graph_rep = torch.cat(feats, -1)
            logits = self.head(graph_rep)

            return logits, {'zz_raw': zz, 'xx_raw': xx, 'zz_edge': zz_edge_attrib, 'xx_edge': xx_edge_attrib}

        return self.head(torch.cat(feats, -1)), {}

def run_interpretability_probe(k=4, dataset='Tox21'):
    QF, AT, AR, Y, SCAF = featurize(k, [dataset])
    model = GraphGInterpret(k, entangler='graph', readout='graph')

    adj = AT
    QFt, At, Yt = torch.tensor(QF[:4]), torch.tensor(adj[:4]), torch.tensor(Y[:4])

    # We do a forward pass to extract edge attributions
    logits, interpret_data = model.forward_with_interpretability(QFt, At)

    print(f"Logits shape: {logits.shape}")
    print(f"ZZ Raw Correlations shape: {interpret_data['zz_raw'].shape}")
    print(f"ZZ Edge Attributions shape (weighted by A_ij): {interpret_data['zz_edge'].shape}")

    # Let's see the first molecule's Z correlations
    print("\nMolecule 0 Edge Attributions (ZZ):")
    for pidx, (i, j) in enumerate(zip(model.pi.tolist(), model.pj.tolist())):
        print(f"Edge ({i},{j}): Adj = {At[0, i, j]:.2f}, Raw <ZZ> = {interpret_data['zz_raw'][0, pidx]:.3f}, Contrib = {interpret_data['zz_edge'][0, pidx]:.3f}")

if __name__ == '__main__':
    run_interpretability_probe(k=4, dataset='Tox21')
