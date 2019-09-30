#coding=utf-8

import numpy as np
from numpy import ndarray, exp, pi, sqrt
from random import randint, seed, random
from numpy.random import choice, seed
from collections import Counter

# 加载肺癌数据集
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
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

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

class GaussianNB(object):
    # 初始化、存储先验概率、训练集的均值、方差及label的类别数量
    def __init__(self):
        self.prior = None
        self.avgs = None
        self.vars = None
        self.n_class = None

    # 计算先验概率
    # 通过Python自带的Counter计算每个类别的占比，再将结果存储到numpy数组中
    def get_prior(self, label):
        cnt = Counter(label)
        prior = np.array([cnt[i] / len(label) for i in range(len(cnt))])
        return prior

    # 计算训练集均值
    # 每个label类别分别计算均值
    def get_avgs(self, data, label):
        return np.array([data[label == i].mean(axis=0) for i in range(self.n_class)])

    # 计算训练集方差
    def get_vasrs(self, data, label):
        return np.array([data[label == i].var(axis=0) for i in range(self.n_class)])

    # 计算似然度
    # 通过高斯分布的概率密度函数计算出似然再连乘得到似然度
    # .prod代表连乘操作
    def get_likehood(self, row):
        return (1 / sqrt(2 * pi * self.vars) * exp(
            -(row - self.avgs) ** 2 / (2 * self.vars))).prod(axis=1)

    # 训练模型
    def fit(self, data, label):
        self.prior = self.get_prior(label)
        self.n_class = len(self.prior)
        self.avgs = self.get_avgs(data, label)
        self.vars = self.get_vasrs(data, label)

    # 预测概率prob
    # 用先验概率乘以似然度再归一化得到每个label的prob
    def predict_prob(self, data):
        likehood = np.apply_along_axis(self.get_likehood, axis=1, arr=data)
        probs = self.prior * likehood
        probs_sum = probs.sum(axis=1)
        return probs / probs_sum[:, None]

    # 预测label
    # 对于单个样本，取prob最大值所对应的类别，就是label的预测值。
    def predict(self, data):
        return self.predict_prob(data).argmax(axis=1)


# 效果评估
def main():
    print("Tesing the performance of Gaussian NaiveBayes...")
    data, label = load_data()
    data_train, data_test, label_train, label_test = train_test_split(data, label, random_state=100)
    clf = GaussianNB()
    clf.fit(data_train, label_train)
    y_hat = clf.predict(data_test)
    acc = get_acc(label_test, y_hat)
    print("Accuracy is %.3f" % acc)


main()
