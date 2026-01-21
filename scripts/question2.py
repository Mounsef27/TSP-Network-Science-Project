import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
DATA_PATH = '/home/ubuntu/Videos/PROJET nsgl/fb100/data'
OUTPUT_DIR = os.path.join(DATA_PATH, 'plots_q2')
FILENAMES = ['Caltech36.gml', 'MIT8.gml', 'Johns Hopkins55.gml']

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def analyze_network(filename):
    filepath = os.path.join(DATA_PATH, filename)
    if not os.path.exists(filepath): return

    short_name = filename.replace('.gml', '')
    print(f"\n--- ANALYSE : {short_name} ---")
    G_raw = nx.read_gml(filepath)
    G = G_raw.subgraph(max(nx.connected_components(G_raw), key=len)).copy()
    
    num_nodes = G.number_of_nodes()
    
    # 1. Distribution Degrés
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(degrees, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Dist. Degrés - {short_name}")
    
    plt.subplot(1, 2, 2)
    degree_counts = np.bincount(degrees)
    k = np.nonzero(degree_counts)[0]
    p_k = degree_counts[k] / num_nodes
    plt.loglog(k, p_k, 'o', markersize=4, color='darkblue', alpha=0.7)
    plt.title(f"Log-Log Dist. - {short_name}")
    plt.xlabel("k (log)"); plt.ylabel("P(k) (log)")
    plt.grid(True, which="both", alpha=0.2)
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"degree_{short_name}.png"), dpi=300)
    plt.close()

    # 2. Métriques
    density = nx.density(G)
    glob_clus = nx.transitivity(G)
    loc_clus = nx.average_clustering(G)
    print(f"Densité: {density:.4f} | Clustering Global: {glob_clus:.4f} | Clustering Local: {loc_clus:.4f}")

    # 3. Scatter Plot (Degré vs Clustering)
    clus_coeffs = nx.clustering(G)
    x = [G.degree(n) for n in G.nodes()]
    y = [clus_coeffs[n] for n in G.nodes()]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=10, alpha=0.5, c='purple')
    plt.title(f"Degré vs Clustering - {short_name}")
    plt.xlabel("Degré k"); plt.ylabel("C(k)")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_{short_name}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    for fname in FILENAMES:
        analyze_network(fname)