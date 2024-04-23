class OneRuleClassifier:
    def __init__(self):
        self.rule = None
        self.target_attribute = None

    def __str__(self):
        return 'My OneRuleClassifier'

    def fit(self, X, y):
        assert len(X) == len(y), "Number of instances and labels must be equal"

        best_rule_accuracy = 0
        self.target_attribute = y[0]

        for feature_index in range(len(X[0])):
            unique_values = set([instance[feature_index] for instance in X])
            for value in unique_values:
                rule = (feature_index, value)
                accuracy = self.calculate_accuracy(X, y, rule)
                if accuracy > best_rule_accuracy:
                    best_rule_accuracy = accuracy
                    self.rule = rule

    def calculate_accuracy(self, X, y, rule):
        correct = 0
        for instance, label in zip(X, y):
            if instance[rule[0]] == rule[1] and label == self.target_attribute:
                correct += 1
        return correct / len(X)

    def predict(self, X):
        assert self.rule is not None, "Classifier has not been trained"
        return [self.target_attribute if instance[self.rule[0]] == self.rule[1] else "other" for instance in X]
