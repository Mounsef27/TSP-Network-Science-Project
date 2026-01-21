import networkx as nx
import torch
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
DATA_PATH = '/home/ubuntu/Videos/PROJET nsgl/fb100/data'
FILENAME = 'Caltech36.gml'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LabelPropagation:
    def __init__(self, adj):
        self.adj = adj.to(DEVICE)
        deg = self.adj.sum(dim=1); deg[deg==0] = 1
        D_inv = torch.diag(1.0/deg)
        self.T = torch.mm(D_inv, self.adj)
        
    def fit_predict(self, labels, mask, max_iter=200):
        n, k = len(labels), len(torch.unique(labels))
        Y = torch.zeros(n, k, device=DEVICE)
        known_idx = mask.nonzero(as_tuple=True)[0]
        Y[known_idx, labels[mask]] = 1.0
        Y_static = Y.clone()
        
        for _ in range(max_iter):
            Y = torch.mm(self.T, Y)
            Y[known_idx] = Y_static[known_idx] # Clamp
        return torch.argmax(Y, dim=1)

if __name__ == "__main__":
    path = os.path.join(DATA_PATH, FILENAME)
    if os.path.exists(path):
        G = nx.read_gml(path)
        G = nx.convert_node_labels_to_integers(G)
        adj = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
        lp = LabelPropagation(adj)
        
        res = []
        for attr in ['dorm', 'year', 'gender', 'major_index']:
            raw = [G.nodes[n].get(attr, 0) for n in range(len(G))]
            y = torch.tensor(LabelEncoder().fit_transform(raw), dtype=torch.long, device=DEVICE)
            
            for f in [0.1, 0.2, 0.3]:
                mask = (torch.rand(len(G)) > f).to(DEVICE)
                pred = lp.fit_predict(y, mask)
                
                # Eval sur inconnus
                idx_test = (~mask).nonzero(as_tuple=True)[0]
                acc = accuracy_score(y[idx_test].cpu(), pred[idx_test].cpu())
                res.append({'Attribute': attr, 'Missing': f, 'Accuracy': acc})
                print(f"{attr} (Missing {f}): Acc={acc:.4f}")
        
        pd.DataFrame(res).to_csv(os.path.join(DATA_PATH, 'results_q5.csv'), index=False)