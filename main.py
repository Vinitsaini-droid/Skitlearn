from Models.linear_regression import LinearRegression
from Models.polynomial_regression import PolynomialRegression
from Models.logistic_regression import LogisticRegression
from Models.neural_network import NeuralNetwork
from models.decision_tree import DecisionTreeClassifier
from models.random_forest import RandomForestClassifier
from Core.metrics import accuracy, precision, recall, f1_score
from Core.metrics import mean_squared_error, log_loss
from sklearn.model_selection import train_test_split
from models.knn import KNNClassifier
from models import KMeans
from models.pca import PCA
from Core.metrics import inertia
from sklearn.datasets import load_iris
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

# Dummy use case for Neural Network
nn = NeuralNetwork(layer_sizes=[10, 16, 3], task='multiclass')
nn.fit(X_train, y_train)  # y should be integer class labels
y_pred = nn.predict(X_test)


# Dummy toy dataset (AND logic gate)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])  # Output is 1 only when both inputs are 1

# Train/test split (manual for simplicity)
X_train, y_train = X[:3], y[:3]
X_test, y_test = X[3:], y[3:]

print("===== Decision Tree Classifier =====")
dt = DecisionTreeClassifier(max_depth=2)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
print("Predictions:", dt_preds)
print("Accuracy:", accuracy(y_test, dt_preds))
print("Precision:", precision(y_test, dt_preds))
print("Recall:", recall(y_test, dt_preds))
print("F1 Score:", f1_score(y_test, dt_preds))

print("\n===== Random Forest Classifier =====")
rf = RandomForestClassifier(n_estimators=5, max_depth=2)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Predictions:", rf_preds)
print("Accuracy:", accuracy(y_test, rf_preds))
print("Precision:", precision(y_test, rf_preds))
print("Recall:", recall(y_test, rf_preds))
print("F1 Score:", f1_score(y_test, rf_preds))

print("\nKNNclassifier")
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
print("KNN Accuracy:", knn.score(X_test, y_test))


print("\nPCA")
iris = load_iris()
X = iris.data
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)


# Simple 2D dataset
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
# Train KMeans
model = KMeans(n_clusters=2, random_state=42)
labels = model.fit_predict(X)
print("Cluster labels:", labels)
print("Centroids:\n", model.centroids)
score = inertia(X, labels, model.centroids)
print("Inertia:", score)
