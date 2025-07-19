import numpy as np
from .linear_regression import LinearRegression

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.linear_regression = LinearRegression()
        self.X_poly = None

    def _polynomial_features(self, X):
        X = np.array(X)
        X_poly = np.ones((X.shape[0], 1))
        for power in range(1, self.degree + 1):
            X_poly = np.c_[X_poly, X ** power]
        return X_poly

    def fit(self, X, y):
        self.X_poly = self._polynomial_features(X)
        self.linear_regression.fit(self.X_poly, y)

    def predict(self, X):
        X_poly = self._polynomial_features(X)
        return self.linear_regression.predict(X_poly)
