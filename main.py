# skitlearn/__main__.py

from skitlearn.models.linear_regression import LinearRegression
from skitlearn.models.polynomial_regression import PolynomialRegression
from skitlearn.core.preprocessing import PolynomialFeatures
import numpy as np

# Example for Linear Regression
print("===== Linear Regression Demo =====")
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
lr = LinearRegression()
lr.fit(X, y)
print("Linear Predictions:", lr.predict(X))

# Example for Polynomial Regression
print("\n===== Polynomial Regression Demo =====")
X_poly = np.array([[1], [2], [3], [4]])
y_poly = np.array([1, 4, 9, 16])  # Quadratic relationship

pr = PolynomialRegression(degree=2)
pr.fit(X_poly, y_poly)
print("Polynomial Predictions:", pr.predict(X_poly))
