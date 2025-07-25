import numpy as np
from models import KMeans
from metrics import inertia
from preprocessing import StandardScaler

def test_kmeans_basic():
    # Basic clustered data
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=2, random_state=42)
    labels = model.fit_predict(X_scaled)

    assert len(labels) == len(X_scaled), "Number of labels should match number of samples"
    assert model.centroids.shape == (2, X.shape[1]), "Centroids shape incorrect"
    assert inertia(X_scaled, labels, model.centroids) >= 0, "Inertia should be non-negative"

    print("✅ test_kmeans_basic passed.")

if __name__ == "__main__":
    test_kmeans_basic()
