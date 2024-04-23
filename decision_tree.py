import numpy as np


class DecisionTreeClassifier:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = {}

    def __str__(self):
        return 'My DecisionTreeClassifier'

    def fit(self, X, y, depth=0):
        self.tree = self._fit(X, y, depth)

    def _fit(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            return {'class': y[0]}
        if self.max_depth is not None and depth >= self.max_depth:
            return {'class': np.bincount(y).argmax()}

        best_split = self.find_best_split(X, y)

        if best_split is None:
            return {'class': np.bincount(y).argmax()}

        feature_index, threshold = best_split
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]

        decision_node = {
            'feature_index': feature_index,
            'threshold': threshold,
            'left': self._fit(X[left_indices], y[left_indices], depth + 1),
            'right': self._fit(X[right_indices], y[right_indices], depth + 1)
        }

        return decision_node

    def find_best_split(self, X, y):
        best_gini = float('inf')
        best_split = None

        for feature_index in range(X.shape[1]):
            feature_values = np.unique(X[:, feature_index])
            for value in feature_values:
                left_indices = np.where(X[:, feature_index] <= value)[0]
                right_indices = np.where(X[:, feature_index] > value)[0]

                gini = self.calculate_gini_index(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, value)

        return best_split

    def calculate_gini_index(self, left_labels, right_labels):
        total_instances = len(left_labels) + len(right_labels)
        gini_left = self.calculate_gini_impurity(left_labels)
        gini_right = self.calculate_gini_impurity(right_labels)
        gini_index = (len(left_labels) / total_instances) * gini_left + (
                len(right_labels) / total_instances) * gini_right
        return gini_index

    def calculate_gini_impurity(self, labels):
        if len(labels) == 0:
            return 0
        counts = np.bincount(labels)
        probabilities = counts / len(labels)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def predict_instance(self, instance, tree):
        if 'class' in tree:
            return tree['class']
        if instance[tree['feature_index']] <= tree['threshold']:
            return self.predict_instance(instance, tree['left'])
        else:
            return self.predict_instance(instance, tree['right'])

    def predict(self, X):
        return [self.predict_instance(instance, self.tree) for instance in X]
