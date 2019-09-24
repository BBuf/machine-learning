#coding=utf-8
import numpy as np
from numpy.random import choice, seed
from random import sample, normalvariate
from numpy import ndarray
from time import time
from random import randint, seed, random
from math import exp

# 统计程序运行时间函数
# fn代表运行的函数
def run_time(fn):
    def fun():
        start = time()
        fn()
        ret = time() - start
        if ret < 1e-6:
            unit = "ns"
            ret *= 1e9
        elif ret < 1e-3:
            unit = "us"
            ret *= 1e6
        elif ret < 1:
            unit = "ms"
            ret *= 1e3
        else:
            unit = "s"
        print("Total run time is %.1f %s\n" % (ret, unit))
    return fun()

def load_data():
    f = open("boston/breast_cancer.csv")
    X = []
    y = []
    for line in f:
        line = line[:-1].split(',')
        xi = [float(s) for s in line[:-1]]
        yi = line[-1]
        if '.' in yi:
            yi = float(yi)
        else:
            yi = int(yi)
        X.append(xi)
        y.append(yi)
    f.close()
    return X, y

# 划分训练集和测试集
def train_test_split(X, y, prob=0.7, random_state=None):
    if random_state is not None:
        seed(random_state)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(len(X)):
        if random() < prob:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])
    seed()
    return X_train, X_test, y_train, y_test


# 将数据归一化到[0, 1]范围
def min_max_scale(X):
    m = len(X[0])
    x_max = [-float('inf') for _ in range(m)]
    x_min = [float('inf') for _ in range(m)]
    for row in X:
        x_max = [max(a, b) for a, b in zip(x_max, row)]
        x_min = [min(a, b) for a, b in zip(x_min, row)]

    ret = []
    for row in X:
        tmp = [(x - b) / (a - b) for a, b, x in zip(x_max, x_min, row)]
        ret.append(tmp)
    return ret

# 准确率
def get_acc(y, y_hat):
    return sum(yi == yi_hat for yi, yi_hat in zip(y, y_hat)) / len(y)

# 查准率
def get_precision(y, y_hat):
    true_postive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    predicted_positive = sum(y_hat)
    return true_postive / predicted_positive

# 查全率
def get_recall(y, y_hat):
    true_postive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    actual_positive = sum(y)
    return true_postive / actual_positive

# 计算真正率
def get_tpr(y, y_hat):
    true_positive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    actual_positive = sum(y)
    return true_positive / actual_positive

# 计算真负率
def get_tnr(y, y_hat):
    true_negative = sum(1 - (yi or yi_hat) for yi, yi_hat in zip(y, y_hat))
    actual_negative = len(y) - sum(y)
    return true_negative / actual_negative

# 画ROC曲线
def get_roc(y, y_hat_prob):
    thresholds = sorted(set(y_hat_prob), reverse=True)
    ret = [[0, 0]]
    for threshold in thresholds:
        y_hat = [int(yi_hat_prob >= threshold) for yi_hat_prob in y_hat_prob]
        ret.append([get_tpr(y, y_hat), 1 - get_tnr(y, y_hat)])
    return ret
# 计算AUC(ROC曲线下方的面积)
def get_auc(y, y_hat_prob):
    roc = iter(get_roc(y, y_hat_prob))
    tpr_pre, fpr_pre = next(roc)
    auc = 0
    for tpr, fpr in roc:
        auc += (tpr + tpr_pre) * (fpr - fpr_pre) / 2
        tpr_pre = tpr
        fpr_pre = fpr
    return auc

def model_evaluation(clf, X, y):
    y_hat = clf.predict(X)
    y_hat_prob = [clf._predict(Xi) for Xi in X]
    ret = dict()
    ret["Accuracy"] = get_acc(y, y_hat)
    ret["Recall"] = get_recall(y, y_hat)
    ret['Precision'] = get_precision(y, y_hat)
    ret['AUC'] = get_auc(y, y_hat_prob)
    for k, v in ret.items():
        print("%s: %.3f" % (k, v))
    print()
    return ret

def sigmoid(x, x_min=-100):
    return 1 / (1 + exp(-x)) if x > x_min else 0

class RegressionBase(object):
    def __init__(self):
        self.bias = None
        self.weights = None

    def _predict(self, Xi):
        return NotImplemented

    def get_gradient_delta(self, Xi, yi):
        y_hat = self._predict(Xi)
        bias_grad_delta = yi - y_hat
        weights_grad_delta = [bias_grad_delta * Xij for Xij in Xi]
        return bias_grad_delta, weights_grad_delta

    # 全梯度下降
    def batch_gradient_descent(self, X, y, lr, epochs):
        #b = b - learning_rate * 1 / m * b_grad_i, b_grad_i < - grad
        #W = W - learning_rate * 1 / m * w_grad_i, w_grad_i < - grad
        m, n = len(X), len(X[0])
        self.bias = 0
        # 正太分布
        self.weights = [normalvariate(0, 0.01) for _ in range(n)]
        for _ in range(epochs):
            bias_grad = 0
            weights_grad = [0 for _ in range(n)]
            for i in range(m):
                bias_grad_delta, weights_grad_delta = self.get_gradient_delta(X[i], y[i])
                bias_grad += bias_grad_delta
                weights_grad = [w_grad + w_grad_d for w_grad, w_grad_d
                                in zip(weights_grad, weights_grad_delta)]
            self.bias = self.bias + lr * bias_grad * 2 / m
            self.weights = [w + lr * w_grad * 2 / m for w,
                                                        w_grad in zip(self.weights, weights_grad)]

    # 随机梯度下降
    def stochastic_gradient_descent(self, X, y, lr, epochs, sample_rate):
        m, n = len(X), len(X[0])
        k = int(m * sample_rate)
        self.bias = 0
        self.weights = [normalvariate(0, 0.01) for _ in range(n)]
        for _ in range(epochs):
            for i in sample(range(m), k):
                bias_grad, weights_grad = self.get_gradient_delta(X[i], y[i])
                self.bias += lr * bias_grad
                self.weights = [w + lr * w_grad for w,
                                                    w_grad in zip(self.weights, weights_grad)]

    def fit(self, X, y, lr, epochs, method="batch", sample_rate=1.0):
        assert method in ("batch", "stochastic")
        if method == "batch":
            self.batch_gradient_descent(X, y, lr, epochs)
        if method == "stochastic":
            self.stochastic_gradient_descent(X, y, lr, epochs, sample_rate)

    def predict(self, X):
        return NotImplemented

class LogisticRegression(RegressionBase):
    def __init__(self):
        RegressionBase.__init__(self)

    def _predict(self, Xi):
        z = sum(wi * xij for wi, xij in zip(self.weights, Xi)) + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return [int(self._predict(Xi) >= threshold) for Xi in X]


def main():
    # Load data
    X, y = load_data()
    X = min_max_scale(X)
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    @run_time
    def batch():
        print("Tesing the performance of LogisticRegression(batch)...")
        # Train model
        clf = LogisticRegression()
        clf.fit(X=X_train, y=y_train, lr=0.05, epochs=200)
        # Model evaluation
        model_evaluation(clf, X_test, y_test)

    @run_time
    def stochastic():
        print("Tesing the performance of LogisticRegression(stochastic)...")
        # Train model
        clf = LogisticRegression()
        clf.fit(X=X_train, y=y_train, lr=0.01, epochs=200,
                method="stochastic", sample_rate=0.5)
        # Model evaluation
        model_evaluation(clf, X_test, y_test)

main()
