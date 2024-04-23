import numpy as np


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def __str__(self):
        return 'My KNNClassifier'

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for instance in X_test:
            distances = [np.linalg.norm(instance - x) for x in self.X_train]
            sorted_indices = np.argsort(distances)
            nearest_neighbors = [self.y_train[i] for i in sorted_indices[:self.k]]
            prediction = max(set(nearest_neighbors), key=nearest_neighbors.count)
            predictions.append(prediction)
        return predictions
