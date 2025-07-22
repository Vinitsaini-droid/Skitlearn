import numpy as np
from collections import Counter
from metrics import entropy

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.root = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_input(x, self.root) for x in X])

    def _predict_input(self, x, node):
        while not node.is_leaf():
            if x[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if (depth == self.max_depth) or (n_samples < self.min_samples_split) or len(set(y)) == 1:
            return Node(value=self._majority_class(y))

        best_feat, best_thresh = self._best_split(X, y, n_features)
        if best_feat is None:
            return Node(value=self._majority_class(y))

        left_indices = X[:, best_feat] < best_thresh
        right_indices = ~left_indices

        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None
        parent_entropy = entropy(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = y[X[:, feature] < t]
                right = y[X[:, feature] >= t]

                if len(left) == 0 or len(right) == 0:
                    continue

                gain = self._information_gain(y, left, right, parent_entropy)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = t

        return split_idx, split_thresh

    def _information_gain(self, parent, left, right, parent_entropy):
        n = len(parent)
        n_l, n_r = len(left), len(right)
        if n_l == 0 or n_r == 0:
            return 0
        weighted_entropy = (n_l / n) * entropy(left) + (n_r / n) * entropy(right)
        return parent_entropy - weighted_entropy

    def _majority_class(self, y):
        return Counter(y).most_common(1)[0][0]
