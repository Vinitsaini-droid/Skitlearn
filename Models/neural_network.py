import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes=[4, 8, 3], learning_rate=0.01, epochs=1000, batch_size=32, seed=42, task='binary'):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.task = task
        np.random.seed(seed)
        self.params = {}

    def _init_weights(self):
        self.params = {}
        for i in range(1, len(self.layer_sizes)):
            self.params[f"W{i}"] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * 0.01
            self.params[f"b{i}"] = np.zeros((self.layer_sizes[i], 1))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return z > 0

    def _softmax(self, z):
        z -= np.max(z, axis=0, keepdims=True)  # For numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def _forward(self, X):
        cache = {'A0': X.T}
        A = X.T
        for i in range(1, len(self.layer_sizes)):
            Z = self.params[f"W{i}"] @ A + self.params[f"b{i}"]
            if i == len(self.layer_sizes) - 1:
                A = self._softmax(Z) if self.task == 'multiclass' else self._sigmoid(Z)
            else:
                A = self._relu(Z)
            cache[f"Z{i}"], cache[f"A{i}"] = Z, A
        return A, cache

    def _backward(self, y, cache):
        grads = {}
        m = y.shape[0]
        y = y.T
        L = len(self.layer_sizes) - 1
        A_final = cache[f"A{L}"]

        if self.task == 'multiclass':
            dZ = A_final - y
        else:
            dZ = A_final - y

        for i in reversed(range(1, L + 1)):
            A_prev = cache[f"A{i-1}"]
            grads[f"dW{i}"] = dZ @ A_prev.T / m
            grads[f"db{i}"] = np.sum(dZ, axis=1, keepdims=True) / m
            if i > 1:
                dA_prev = self.params[f"W{i}"].T @ dZ
                dZ = dA_prev * self._relu_derivative(cache[f"Z{i-1}"])
        
        return grads

    def _update_params(self, grads):
        for i in range(1, len(self.layer_sizes)):
            self.params[f"W{i}"] -= self.lr * grads[f"dW{i}"]
            self.params[f"b{i}"] -= self.lr * grads[f"db{i}"]

    def fit(self, X, y):
        if self.task == 'multiclass':
            num_classes = self.layer_sizes[-1]
            y = np.eye(num_classes)[y]  # One-hot encode

        self._init_weights()
        for epoch in range(self.epochs):
            A_final, cache = self._forward(X)
            grads = self._backward(y, cache)
            self._update_params(grads)

    def predict_proba(self, X):
        A_final, _ = self._forward(X)
        return A_final.T

    def predict(self, X):
        probs = self.predict_proba(X)
        if self.task == 'multiclass':
            return np.argmax(probs, axis=1)
        else:
            return (probs >= 0.5).astype(int)
