#coding=utf-8
import numpy as np
from numpy.random import choice, seed
from random import sample, normalvariate
from numpy import ndarray
from time import time
from random import randint, seed, random

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
    f = open("boston/housing.csv")
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

# 计算回归模型的拟合优度
def get_r2(reg, X, y):
    y_hat = reg.predict(X)
    m = len(y)
    n = len(y_hat)
    sse = sum((yi - yi_hat) ** 2 for yi, yi_hat in zip(y, y_hat))
    y_avg = sum(y) / len(y)
    sst = sum((yi - y_avg) ** 2 for yi in y)
    r2 = 1 - sse / sst
    print("Test r2 is %.3f!" % r2)
    return r2

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


class LinearRegreession(RegressionBase):
    def __init__(self):
        RegressionBase.__init__(self)

    def _predict(self, Xi):
        return sum(wi * xij for wi, xij in zip(self.weights, Xi)) + self.bias

    def predict(self, X):
        return [self._predict(xi) for xi in X]



def main():
    X, y = load_data()
    X = min_max_scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    @run_time
    def batch():
        print("Tesing the performance of LinearRegression(batch)...")
        reg = LinearRegreession()
        reg.fit(X=X_train, y=y_train, lr=0.02, epochs=5000)
        get_r2(reg, X_test, y_test)

    @run_time
    def stochastic():
        print("Tesing the performance of LinearRegression(stochastic)...")
        reg = LinearRegreession()
        reg.fit(X=X_train, y=y_train, lr=0.001, epochs=1000,
                method="stochastic", sample_rate=0.5)
        get_r2(reg, X_test, y_test)


main()
