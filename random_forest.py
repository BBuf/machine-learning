#coding=utf-8
import decision_tree
from decision_tree import DecisionTree
from random import sample, choices, choice

class RandomForest(object):
    def __init__(self):
        self.trees = None
        self.tree_features = None

    def fit(self, X, y, n_estimators=10, max_depth=3, min_samples_split=2, max_features=None, n_samples=None):
        self.trees = []
        self.tree_features = []
        for _ in range(n_estimators):
            m = len(X[0])
            n = len(y)
            if n_samples:
                idx = choices(population=range(n), k=min(n, n_samples))
            else:
                idx = range(n)
            if max_features:
                n_features = min(m, max_features)
            else:
                n_features = int(m ** 0.5)
            features = sample(range(m), choice(range(1, n_features+1)))
            X_sub = [[X[i][j] for j in features] for i in idx]
            y_sub = [y[i] for i in idx]
            clf = DecisionTree()
            clf.fit(X_sub, y_sub, max_depth, min_samples_split)
            self.trees.append(clf)
            self.tree_features.append(features)

    def _predict(self, Xi):
        pos_vote = 0
        for tree, features in zip(self.trees, self.tree_features):
            score = tree._predict([Xi[j] for j in features])
            if score >= 0.5:
                pos_vote += 1
        neg_vote = len(self.trees) - pos_vote
        if pos_vote > neg_vote:
            return 1
        elif pos_vote < neg_vote:
            return 0
        else:
            return choice([0, 1])

    def predict(self, X):
        return [self._predict(Xi) for Xi in X]


@decision_tree.run_time
def main():
    print("Tesing the performance of RandomForest...")
    # Load data
    X, y = decision_tree.load_data()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = decision_tree.train_test_split(X, y, random_state=40)

    # Train model
    rf = RandomForest()
    rf.fit(X_train, y_train, n_samples=300, max_depth=3, n_estimators=20)
    # Model evaluation
    y_hat = rf.predict(X_test)
    acc = decision_tree.get_acc(y_test, y_hat)
    print("Accuracy is %.3f" % acc)