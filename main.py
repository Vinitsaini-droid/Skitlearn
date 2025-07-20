from Models.linear_regression import LinearRegression
from Models.polynomial_regression import PolynomialRegression
from Models.logistic_regression import LogisticRegression
from Core.preprocessing import mean_squared_error, log_loss

import numpy as np

# Dummy data for Linear Regression
X_lin = np.array([[1], [2], [3], [4]])
y_lin = np.array([2, 4, 6, 8])

lr = LinearRegression()
lr.fit(X_lin, y_lin)
y_pred_lin = lr.predict(X_lin)
print("Linear Regression Predictions:", y_pred_lin)
print("MSE:", mean_squared_error(y_lin, y_pred_lin))

# Dummy data for Polynomial Regression
pr = PolynomialRegression(degree=2)
pr.fit(X_lin, y_lin)
y_pred_poly = pr.predict(X_lin)
print("Polynomial Regression Predictions:", y_pred_poly)
print("MSE (Poly):", mean_squared_error(y_lin, y_pred_poly))

# Dummy data for Logistic Regression
X_log = np.array([[0], [1], [2], [3], [4]])
y_log = np.array([0, 0, 0, 1, 1])

logr = LogisticRegression(learning_rate=0.1, n_iterations=1000)
logr.fit(X_log, y_log)
y_pred_log_proba = logr.predict_proba(X_log)
y_pred_log = logr.predict(X_log)

print("Logistic Regression Probabilities:", y_pred_log_proba)
print("Logistic Regression Predictions:", y_pred_log)
print("Log Loss:", log_loss(y_log, y_pred_log_proba))
