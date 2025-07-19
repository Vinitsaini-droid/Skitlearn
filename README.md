## 📘 Models Supported

- ✅ Linear Regression (Simple + Multiple)
- ✅ Polynomial Regression (with custom degree)
- 🚧 More coming soon: Logistic Regression, Decision Trees, etc.

## ✨ Example Usage

```python
from skitlearn.models.polynomial_regression import PolynomialRegression
import numpy as np

X = np.array([[1], [2], [3]])
y = np.array([1, 4, 9])  # y = x^2

model = PolynomialRegression(degree=2)
model.fit(X, y)
preds = model.predict(X)
print(preds)
