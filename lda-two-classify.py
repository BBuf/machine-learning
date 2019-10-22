#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification

def LDA(X, y):
    X1 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
    X2 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

    len1 = len(X1)
    len2 = len(X2)

    # 求均值向量u1,u2
    miu1 = np.mean(X1, axis=0)
    miu2 = np.mean(X2, axis=0)

    # 求S_w
    # \sum_0
    conv1 = np.dot((X1 - miu1).T, (X1 - miu1))
    # \sum_1
    conv2 = np.dot((X2 - miu2).T, (X2 - miu2))
    Sw = conv1 + conv2

    # 计算w
    w = np.dot(np.mat(Sw).I, (miu1 - miu2).reshape((len(miu1), 1)))
    X1_new = np.dot(X1, w)
    X2_new = np.dot(X2, w)
    y1_new = [0 for i in range(len1)]
    y2_new = [1 for i in range(len2)]
    return X1_new, X2_new, y1_new, y2_new

def main():
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_classes=2,
                               n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)

    X1_new, X2_new, y1_new, y2_new = LDA(X, y)

    # 可视化原始数据
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.show()
    # 可视化LDA降维后的数据
    plt.plot(X1_new, y1_new, "bo")
    plt.plot(X2_new, y2_new, "ro")
    plt.show()

main()

