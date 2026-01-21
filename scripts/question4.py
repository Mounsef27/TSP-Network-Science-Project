import networkx as nx
import random
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# --- CONFIGURATION ---
DATA_PATH = '/home/ubuntu/Videos/PROJET nsgl/fb100/data'
OUTPUT_DIR = os.path.join(DATA_PATH, 'plots_q4')
TARGETS = ['Caltech36.gml', 'Reed98.gml', 'Bucknell39.gml', 'Rice31.gml', 'Vassar85.gml']
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

class LinkPrediction(ABC):
    def __init__(self, graph): self.graph = graph
    @abstractmethod
    def fit(self): pass
    @abstractmethod
    def predict(self, pairs): pass

class CommonNeighbors(LinkPrediction):
    def fit(self): self.n_sets = {n: set(self.graph.neighbors(n)) for n in self.graph.nodes()}
    def predict(self, pairs):
        return [(u, v, len(self.n_sets[u].intersection(self.n_sets[v]))) 
                if u in self.n_sets and v in self.n_sets else (u,v,0) for u,v in pairs]

class Jaccard(LinkPrediction):
    def fit(self): self.n_sets = {n: set(self.graph.neighbors(n)) for n in self.graph.nodes()}
    def predict(self, pairs):
        res = []
        for u, v in pairs:
            if u in self.n_sets and v in self.n_sets:
                i = len(self.n_sets[u].intersection(self.n_sets[v]))
                u_len = len(self.n_sets[u].union(self.n_sets[v]))
                res.append((u, v, i/u_len if u_len>0 else 0))
            else: res.append((u,v,0))
        return res

class AdamicAdar(LinkPrediction):
    def fit(self):
        self.n_sets = {n: set(self.graph.neighbors(n)) for n in self.graph.nodes()}
        self.weights = {n: 1/math.log(len(s)) if len(s)>1 else 0 for n, s in self.n_sets.items()}
    def predict(self, pairs):
        res = []
        for u, v in pairs:
            score = 0
            if u in self.n_sets and v in self.n_sets:
                for z in self.n_sets[u].intersection(self.n_sets[v]):
                    score += self.weights.get(z, 0)
            res.append((u, v, score))
        return res

def evaluate(G, predictor_cls):
    edges = list(G.edges())
    random.shuffle(edges)
    num_rem = int(len(edges)*0.1)
    removed = set(edges[:num_rem])
    G_train = nx.Graph(); G_train.add_nodes_from(G.nodes()); G_train.add_edges_from(edges[num_rem:])
    
    # Candidats: Removed + Noise
    candidates = list(removed)
    while len(candidates) < num_rem * 10:
        u, v = random.sample(list(G.nodes()), 2)
        if u!=v and not G_train.has_edge(u,v): candidates.append((u,v))
    
    model = predictor_cls(G_train); model.fit()
    preds = model.predict(list(set(candidates)))
    preds.sort(key=lambda x: x[2], reverse=True)
    
    # Precision@100
    top100 = preds[:100]
    hits = sum([1 for u,v,_ in top100 if (u,v) in removed or (v,u) in removed])
    return hits / 100.0

if __name__ == "__main__":
    results = []
    for fname in TARGETS:
        path = os.path.join(DATA_PATH, fname)
        if os.path.exists(path):
            G = nx.read_gml(path)
            cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(cc).copy()
            print(f"Graph: {fname}")
            row = {'Graph': fname}
            for name, cls in [('CN', CommonNeighbors), ('Jaccard', Jaccard), ('AA', AdamicAdar)]:
                acc = evaluate(G, cls)
                row[name] = acc
                print(f"  > {name} Prec@100: {acc}")
            results.append(row)
    
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'results_q4.csv'), index=False)