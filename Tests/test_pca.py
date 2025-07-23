import numpy as np
from models.pca import PCA

def test_pca_fit_transform():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    pca = PCA(n_components=1)
    X_reduced = pca.fit_transform(X)
    assert X_reduced.shape == (3, 1)
