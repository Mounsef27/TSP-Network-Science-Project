import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
DATA_PATH = '/home/ubuntu/Videos/PROJET nsgl/fb100/data'
OUTPUT_DIR = os.path.join(DATA_PATH, 'plots_q3')
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

ATTRIBUTES = [
    ('student_fac', 'Student/Faculty Status', 'attribute'),
    ('major_index', 'Major', 'attribute'),
    ('degree', 'Vertex Degree', 'degree'),
    ('dorm', 'Dorm', 'attribute'),
    ('gender', 'Gender', 'attribute')
]

def process_single_graph(filepath):
    res = {'size': 0}
    try:
        G = nx.read_gml(filepath)
        res['size'] = G.number_of_nodes()
        for key, _, method in ATTRIBUTES:
            try:
                if method == 'degree': val = nx.degree_assortativity_coefficient(G)
                else: val = nx.attribute_assortativity_coefficient(G, key)
            except: val = np.nan
            res[key] = val
        return res
    except: return None

if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA_PATH, '*.gml'))
    print(f"Traitement de {len(files)} graphes sur {cpu_count()} cœurs...")
    
    with Pool(cpu_count()) as pool:
        results = pool.map(process_single_graph, files)
    
    # Agrégation
    data = {k[0]: {'sizes': [], 'values': []} for k in ATTRIBUTES}
    for r in results:
        if r:
            for k in data:
                data[k]['sizes'].append(r['size'])
                data[k]['values'].append(r.get(k, np.nan))

    # Plot & Report
    for key, name, _ in ATTRIBUTES:
        sizes = np.array(data[key]['sizes'])
        vals = np.array(data[key]['values'])
        mask = ~np.isnan(vals)
        sizes, vals = sizes[mask], vals[mask]
        
        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].scatter(sizes, vals, alpha=0.6, c='royalblue', edgecolors='k')
        ax[0].set_xscale('log'); ax[0].axhline(0, c='red', ls='--')
        ax[0].set_title(f"{name}: Assortativity vs Size")
        
        ax[1].hist(vals, bins=15, color='skyblue', edgecolor='black', density=True)
        ax[1].axvline(np.mean(vals), c='green', lw=2, label=f"Mean: {np.mean(vals):.3f}")
        ax[1].set_title(f"{name}: Distribution"); ax[1].legend()
        
        plt.savefig(os.path.join(OUTPUT_DIR, f"plot_{key}.png"), dpi=300)
        plt.close()
        
        print(f"{name:<25} | Mean: {np.mean(vals):.4f} | Std: {np.std(vals):.4f}")