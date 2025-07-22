import numpy as np
from models.decision_tree import DecisionTreeClassifier

def test_decision_tree():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert (preds == y).all()
