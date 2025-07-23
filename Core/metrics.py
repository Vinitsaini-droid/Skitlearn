import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def log_loss(y_true, y_pred_proba, eps=1e-15):
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))

def categorical_cross_entropy(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1)
)
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def entropy(y):
    """Calculate entropy of label array y."""
    if len(y) == 0:
        return 0
    proportions = np.bincount(y) / len(y)
    proportions = proportions[proportions > 0]
    return -np.sum(proportions * np.log2(proportions))


def gini(y):
    """Calculate Gini impurity of label array y."""
    if len(y) == 0:
        return 0
    proportions = np.bincount(y) / len(y)
    return 1 - np.sum(proportions ** 2)

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp + 1e-15)


def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-15)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-15)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

