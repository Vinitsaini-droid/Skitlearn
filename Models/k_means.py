import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def _initialize_centroids(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else self.centroids[i]
            for i in range(self.n_clusters)
        ])
        return new_centroids

    def fit(self, X):
        X = np.array(X)
        self.centroids = self._initialize_centroids(X)

        for i in range(self.max_iter):
            labels = self._assign_clusters(X, self.centroids)
            new_centroids = self._update_centroids(X, labels)

            # Check convergence
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

        self.labels_ = self._assign_clusters(X, self.centroids)

    def predict(self, X):
        X = np.array(X)
        return self._assign_clusters(X, self.centroids)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
