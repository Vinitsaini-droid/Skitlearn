import numpy as np
from ml_lib.models import NeuralNetwork

def generate_dummy_binary():
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def generate_dummy_multiclass():
    X = np.random.randn(120, 4)
    y = np.random.randint(0, 3, 120)
    return X, y

def test_binary_fit_and_predict():
    X, y = generate_dummy_binary()
    nn = NeuralNetwork(layer_sizes=[4, 8, 1], epochs=200, learning_rate=0.1, task='binary')
    nn.fit(X, y)
    preds = nn.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1})

def test_multiclass_fit_and_predict():
    X, y = generate_dummy_multiclass()
    nn = NeuralNetwork(layer_sizes=[4, 10, 3], epochs=300, learning_rate=0.1, task='multiclass')
    nn.fit(X, y)
    preds = nn.predict(X)
    assert preds.shape == y.shape
    assert set(preds).issubset(set(y))

def test_predict_proba_shape_binary():
    X, y = generate_dummy_binary()
    nn = NeuralNetwork(layer_sizes=[4, 6, 1], epochs=10, task='binary')
    nn.fit(X, y)
    proba = nn.predict_proba(X)
    assert proba.shape == (len(X), 1)

def test_predict_proba_shape_multiclass():
    X, y = generate_dummy_multiclass()
    nn = NeuralNetwork(layer_sizes=[4, 6, 3], epochs=10, task='multiclass')
    nn.fit(X, y)
    proba = nn.predict_proba(X)
    assert proba.shape == (len(X), 3)
