import numpy as np
from skitlearn.models.linear_regression import LinearRegression

def test_simple_line():
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    
    model = LinearRegression()
    model.fit(X, y)
    
    preds = model.predict(np.array([[4]]))
    assert np.isclose(preds[0], 8), f"Expected 8, got {preds[0]}"
