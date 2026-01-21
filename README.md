# Analysis of the Facebook100 Dataset 🎓

**Course:** NET 4103 - Network Science & Graph Learning  
**Institution:** Télécom SudParis  
**Language:** Python 3.8+

## 📝 Project Overview

This project provides a comprehensive structural analysis of social networks within US universities using the **Facebook100 dataset** (a snapshot of the Facebook network from September 2005).

The goal is to understand how students formed social ties during the early days of social media by applying various Graph Learning and Network Science techniques.

### Key Objectives:
1.  **Topological Analysis:** Analyzing degree distributions (Power Law), clustering coefficients, and small-world properties.
2.  **Assortativity:** Investigating homophily to see if students bond based on shared attributes (Dorm, Major, Gender, Status).
3.  **Link Prediction:** Implementing algorithms (Common Neighbors, Jaccard, Adamic/Adar) to predict missing friendships with high precision.
4.  **Community Detection:** Testing sociologically grounded hypotheses (e.g., "Dorms vs. Class Year" as primary drivers of community formation) using the Louvain algorithm.

## 📂 Repository Structure

```text
NET4103-Facebook100-Analysis/
│
├── data/                  # Contains the .gml files (e.g., Caltech36.gml)
├── plots/                 # Generated visualizations used in the report
│   ├── question1/         # Visualizations of node attributes
│   └── ...
│
├── scripts/               # Standalone Python scripts for each question
│   ├── question1.py       # Attribute Visualization
│   ├── question2.py       # Topology Analysis
│   ├── question3.py       # Assortativity (Parallel processing)
│   ├── question4.py       # Link Prediction (Manual implementation)
│   ├── question5.py       # Label Propagation (PyTorch)
│   └── question6.py       # Community Detection (Louvain)
│
├── notebook/              # Full analysis in a single Jupyter Notebook
│   └── Facebook100_Analysis_Full.ipynb
│
├── requirements.txt       # List of python dependencies
└── README.md              # Project documentation
