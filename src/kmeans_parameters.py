import numpy as np
import pickle

class KMeansParameters:
    """K-Means cluster centers for parameter server"""

    def __init__(self, n_clusters=3, n_features=4, init_data=None, random_state=42):
        self.n_clusters = n_clusters
        self.n_features = n_features
        rng = np.random.RandomState(random_state)
        if init_data is not None:
            # Seed centroids from real data points (k-means++-style seeding).
            # Starting from small random values far from the actual data
            # distribution makes the parameter server converge to a much
            # worse local optimum than a single-machine k-means++ run —
            # seeding from real points closes that gap.
            idx = rng.choice(len(init_data), n_clusters, replace=False)
            self.centers = init_data[idx].copy().astype(float)
        else:
            self.centers = rng.randn(n_clusters, n_features) * 2

    def compute_assignments(self, X):
        """Assign each point to nearest cluster"""
        distances = np.zeros((len(X), self.n_clusters))
        for i, center in enumerate(self.centers):
            distances[:, i] = np.linalg.norm(X - center, axis=1)
        return np.argmin(distances, axis=1)

    def compute_updates(self, X, assignments):
        """Compute new cluster centers from assignments"""
        new_centers = np.zeros_like(self.centers)
        counts = np.zeros(self.n_clusters)

        for i in range(self.n_clusters):
            mask = assignments == i
            if np.any(mask):
                new_centers[i] = X[mask].mean(axis=0)
                counts[i] = np.sum(mask)

        return new_centers, counts

    def update_centers(self, updates, counts, learning_rate=0.5):
        """Update centers with weighted average"""
        total_counts = np.sum(counts)
        if total_counts > 0:
            weights = counts / total_counts
            for i in range(self.n_clusters):
                if counts[i] > 0:
                    self.centers[i] = (1 - learning_rate) * self.centers[i] + learning_rate * updates[i]

    def serialize(self):
        return pickle.dumps({'centers': self.centers, 'n_clusters': self.n_clusters, 'n_features': self.n_features})

    @staticmethod
    def deserialize(data):
        obj = pickle.loads(data)
        params = KMeansParameters(obj['n_clusters'], obj['n_features'])
        params.centers = obj['centers']
        return params

    def inertia(self, X):
        """Compute within-cluster sum of squares"""
        assignments = self.compute_assignments(X)
        total = 0
        for i, center in enumerate(self.centers):
            mask = assignments == i
            if np.any(mask):
                total += np.sum((X[mask] - center) ** 2)
        return total
