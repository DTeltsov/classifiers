class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}
        self.classes = []

    def __str__(self):
        return 'My NaiveBayesClassifier'

    def fit(self, X, y):
        num_instances = len(y)
        self.classes = list(set(y))
        for c in self.classes:
            self.class_probs[c] = sum(1 for label in y if label == c) / num_instances
        for c in self.classes:
            self.feature_probs[c] = {}
            c_indices = [i for i, label in enumerate(y) if label == c]
            for feature_index in range(len(X[0])):
                feature_values = [X[i][feature_index] for i in c_indices]
                unique_values = set(feature_values)
                self.feature_probs[c][feature_index] = {}
                for value in unique_values:
                    self.feature_probs[c][feature_index][value] = feature_values.count(value) / len(c_indices)

    def predict(self, X):
        predictions = []
        for instance in X:
            max_prob = -1
            predicted_class = None
            for c in self.classes:
                class_prob = self.class_probs[c]
                for feature_index, value in enumerate(instance):
                    if value in self.feature_probs[c][feature_index]:
                        class_prob *= self.feature_probs[c][feature_index][value]
                    else:
                        class_prob *= 1 / (len(self.feature_probs[c][feature_index]) + 1)
                if class_prob > max_prob:
                    max_prob = class_prob
                    predicted_class = c
            predictions.append(predicted_class)
        return predictions
