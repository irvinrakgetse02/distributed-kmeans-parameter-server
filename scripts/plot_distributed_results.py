import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin

print("=" * 50)
print("GENERATING DISTRIBUTED IRIS CLUSTERS")
print("=" * 50)

# Load data
df = pd.read_csv('iris_augmented.csv')
X = df.values
print(f"Data shape: {X.shape}")

# Load distributed centers
try:
    centers = np.load('final_cluster_centers.npy')
    print(f"Loaded cluster centers from server: {centers.shape}")

    # Assign points to nearest cluster
    labels = pairwise_distances_argmin(X, centers)

except FileNotFoundError:
    print("Error: final_cluster_centers.npy not found. Run server.py first.")
    exit()

# Visualize (sepal length vs sepal width)
sample_idx = np.random.RandomState(42).choice(len(X), 8000, replace=False)
plt.figure(figsize=(10, 8))
plt.scatter(X[sample_idx, 0], X[sample_idx, 1], c=labels[sample_idx], cmap='viridis', alpha=0.5, s=15)
plt.scatter(centers[:, 0], centers[:, 1],
            marker='X', s=250, c='red', edgecolors='black', linewidths=2, label='Final centroids')
plt.title('K-Means Clustering on Iris Dataset (Distributed)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.tight_layout()
plt.savefig('distributed_clusters.png', dpi=150)
print("Saved visualization to distributed_clusters.png")
