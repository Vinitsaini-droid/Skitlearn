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
