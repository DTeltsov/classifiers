import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from tabulate import tabulate
from decision_tree import DecisionTreeClassifier
from knn import KNNClassifier
from naive import NaiveBayesClassifier
from one_rule import OneRuleClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


def main():
    classifiers = [
        OneRuleClassifier,
        KNNClassifier, KNeighborsClassifier,
        NaiveBayesClassifier, GaussianNB,
        DecisionTreeClassifier, SkDecisionTreeClassifier
    ]
    datasets = [
        np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 2, 1],
            [1, 1, 2, 0],
            [1, 0, 2, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1]
        ]),
        np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 2, 1],
            [1, 1, 2, 1],
            [1, 0, 2, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ]),
        np.array([
            [0, 1, 2, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 2, 1],
            [1, 1, 2, 1],
            [1, 0, 2, 1],
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 1, 1]
        ]),
        np.array([
            [0, 1, 2, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 2, 0],
            [1, 1, 2, 1],
            [1, 0, 2, 1],
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 1, 1]
        ])
    ]
    x_test = [[1, 1, 1]]
    for dataset in datasets:
        x_train = dataset[:, :-1]
        y_train = dataset[:, -1]
        results = []
        print(f'Dataset\n{dataset}')
        for classifier_class in classifiers:
            classifier = classifier_class()
            classifier.fit(x_train, y_train)
            prediction = classifier.predict(x_test)
            x, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
            x_train_accuracy, x_test_accuracy, y_train_accuracy, y_test_accuracy = train_test_split(
                x, y, test_size=0.2, random_state=42
            )
            classifier.fit(x_train_accuracy, y_train_accuracy)
            start_time = time.time()
            y_pred = classifier.predict(x_test_accuracy)
            end_time = time.time()
            results.append([f'{classifier}', prediction, accuracy_score(y_test_accuracy, y_pred), round(end_time - start_time, 5)])
        print(tabulate(results, headers=['classifier', 'prediction', 'accuracy', 'execution_time']))
        print('\n\n')


if __name__ == '__main__':
    main()
