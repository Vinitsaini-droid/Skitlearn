import numpy as np
from skitlearn.models.logistic_regression import LogisticRegression

def test_logistic_regression_simple():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(lr=0.1, n_iters=1000)
    model.fit(X, y)
    preds = model.predict(X)

    assert sum(preds) >= 2, "Model failed to classify correctly"
