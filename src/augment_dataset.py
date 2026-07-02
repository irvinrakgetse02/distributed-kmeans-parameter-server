import pandas as pd
import numpy as np

print("=" * 50)
print("AUGMENTING IRIS DATASET")
print("=" * 50)

# Load Iris dataset
df = pd.read_csv('IRIS.csv')
print(f"Original dataset loaded: {len(df)} samples, 4 features")

# Extract features (kept on their natural scale, matching the reported
# single-machine clustering plot: sepal length ~4-7cm, sepal width ~2-4.5cm)
features = df.drop('class', axis=1)
X = features.values

# Augment to 150,000 samples
n_augmented = 150000
np.random.seed(42)

reps = n_augmented // len(X)
augmented = np.tile(X, (reps, 1))

remaining = n_augmented - len(augmented)
if remaining > 0:
    idx = np.random.choice(len(X), remaining)
    noise = np.random.randn(remaining, X.shape[1]) * 0.1
    augmented = np.vstack([augmented, X[idx] + noise])

# Save augmented data
aug_df = pd.DataFrame(augmented, columns=features.columns)
aug_df.to_csv('iris_augmented.csv', index=False)

print(f"Augmented dataset: {len(aug_df):,} samples")
print(f"Features: {aug_df.shape[1]}")
print(f"Saved to: iris_augmented.csv")
