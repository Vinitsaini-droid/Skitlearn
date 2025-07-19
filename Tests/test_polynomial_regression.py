import numpy as np
from Models.polynomial_regression import PolynomialRegression

def test_polynomial_regression():
    # Quadratic example: y = 1 + 2x + 3x^2
    X = np.array([[0], [1], [2], [3]])
    y = 1 + 2*X.flatten() + 3*X.flatten()**2

    model = PolynomialRegression(degree=2)
    model.fit(X, y)
    preds = model.predict(X)

    print("Predictions:", preds)
    print("Expected:", y)

if __name__ == "__main__":
    test_polynomial_regression()
