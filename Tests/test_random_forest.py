import numpy as np
from models.random_forest import RandomForestClassifier

def test_random_forest():
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1, 0])
    model = RandomForestClassifier(n_estimators=5, max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
