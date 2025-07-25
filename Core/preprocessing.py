import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True):
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    split = int(X.shape[0] * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]

class StandardScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def generate_polynomial_features(X, degree):
    X = np.array(X)
    X_poly = np.ones((X.shape[0], 1))
    for power in range(1, degree + 1):
        X_poly = np.c_[X_poly, X ** power]
    return X_poly

def one_hot_encode(y, num_classes=None):
    y = np.array(y)
    if num_classes is None:
        num_classes = np.max(y) + 1
    return np.eye(num_classes)[y]

def ensure_numeric(X):
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("Decision Trees require numeric input.")



