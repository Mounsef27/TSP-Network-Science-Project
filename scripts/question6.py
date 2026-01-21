import networkx as nx
import community.community_louvain as louvain
import pandas as pd
import os
from sklearn.metrics import adjusted_rand_score

# --- CONFIGURATION ---
DATA_PATH = '/home/ubuntu/Videos/PROJET nsgl/fb100/data'
TARGETS = ['Caltech36.gml', 'Bucknell39.gml', 'Rice31.gml', 'Reed98.gml']

if __name__ == "__main__":
    results = []
    print(f"{'Graph':<15} | {'Attribute':<10} | {'ARI':<10}")
    print("-" * 40)
    
    for fname in TARGETS:
        path = os.path.join(DATA_PATH, fname)
        if os.path.exists(path):
            G = nx.read_gml(path)
            # Louvain
            part = louvain.best_partition(G, random_state=42)
            y_pred = [part[n] for n in G.nodes()]
            
            for attr in ['dorm', 'year', 'major_index', 'gender']:
                true_vals = [G.nodes[n].get(attr, 0) for n in G.nodes()]
                # Filtre les valeurs manquantes (0)
                mask = [i for i, v in enumerate(true_vals) if v != 0]
                if len(mask) > len(G)*0.5:
                    ari = adjusted_rand_score([true_vals[i] for i in mask], [y_pred[i] for i in mask])
                    results.append({'Graph': fname, 'Attr': attr, 'ARI': ari})
                    print(f"{fname:<15} | {attr:<10} | {ari:.4f}")
    
    pd.DataFrame(results).to_csv(os.path.join(DATA_PATH, 'results_q6.csv'), index=False)