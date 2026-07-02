import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time

print("=" * 50)
print("SINGLE MACHINE K-MEANS (IRIS DATASET)")
print("=" * 50)

df = pd.read_csv('iris_augmented.csv')
X = df.values
print(f"Data shape: {X.shape}")
print(f"Features: 4 (sepal length, sepal width, petal length, petal width)")

print("\nTraining K-Means...")
start = time.time()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
end = time.time()

print(f"\n✅ Completed in {end - start:.2f} seconds")
print(f"Inertia: {kmeans.inertia_:.2f}")

with open('comparison.txt', 'w') as f:
    f.write(f"Single Machine: {end - start:.2f}s\n")
    f.write(f"Inertia: {kmeans.inertia_:.2f}\n")

print("Baseline results saved.")
