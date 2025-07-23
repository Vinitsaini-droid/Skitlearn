import numpy as np
from models.knn import KNNClassifier

def test_knn_classifier():
    X = np.array([[1], [3], [5], [7]])
    y = np.array([0, 0, 1, 1])
    model = KNNClassifier(k=3)
    model.fit(X, y)
    assert model.predict([[2]])[0] == 0
