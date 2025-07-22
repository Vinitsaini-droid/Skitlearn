import numpy as np
from models.decision_tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape

        if self.max_features == 'sqrt':
            self.n_sub_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            self.n_sub_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            self.n_sub_features = self.max_features
        else:
            self.n_sub_features = n_features

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            feat_indices = np.random.choice(n_features, self.n_sub_features, replace=False)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, feat_indices], y_sample)
            tree.feature_indices = feat_indices
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([
            tree.predict(X[:, tree.feature_indices]) for tree in self.trees
        ])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)
