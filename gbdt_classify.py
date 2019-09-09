#coding=utf-8
from copy import copy
from random import randint, seed, random
from time import time
from regression import RegressionTree
from gbdt_regressor import GradientBoostingBase
from random import choice
from math import exp, log

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

def load_cancer():
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

class GradientBoostingClassifier(GradientBoostingBase):
    def __init__(self):
        GradientBoostingBase.__init__(self)
        self.fn = sigmoid
    def get_init_val(self, y):
        n = len(y)
        y_sum = sum(y)
        return log((y_sum) / (n - y_sum))
    def get_score(self, idxs, y_hat, residuals):
        numerator = denominator = 0
        for idx in idxs:
            numerator += residuals[idx]
            denominator += y_hat[idx] * (1 - y_hat[idx])

        return numerator / denominator
    def predict(self, X, threshold=0.5):
        return [int(self._predict(Xi) >= threshold) for Xi in X]


@run_time
def main():
    print('Testing the accuracy of GBDT ClassifyTree...')
    X, y = load_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=20)
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train, n_estimators=2,
            lr=0.8, max_depth=3, min_samples_split=2)
    model_evaluation(clf, X_test, y_test)

