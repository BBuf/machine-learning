#coding=utf-8
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
# 这是sklearn中实现的LDA，待会我们会比较自己实现的LDA和它的区别
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# k为目标
def LDA(X, y, k):
    label_ = list(set(y))
    X_classify = {}
    for label in label_:
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == label])
        X_classify[label] = X1

    miu = np.mean(X, axis=0)
    miu_classify = {}
    for label in label_:
        miu1 = np.mean(X_classify[label], axis=0)
        miu_classify[label] = miu1

    # St = np.dot((X - mju).T, X - mju)
    # 计算类内散度矩阵Sw
    Sw = np.zeros((len(miu), len(miu)))
    for i in label_:
        Sw += np.dot((X_classify[i] - miu_classify[i]).T, X_classify[i] - miu_classify[i])

    #Sb = St-Sw
    # 计算类内散度矩阵Sb
    Sb = np.zeros((len(miu), len(miu)))
    for i in label_:
        Sb += len(X_classify[i]) * np.dot((miu_classify[i] - miu).reshape(
            (len(miu), 1)), (miu_classify[i] - miu).reshape((1, len(miu))))

    # 计算S_w^{-1}S_b的特征值和特征矩阵
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    sorted_indices = np.argsort(eig_vals)
    # 提取前k个特征向量
    topk_eig_vecs = eig_vecs[:, sorted_indices[:-k - 1:-1]]
    return topk_eig_vecs

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    W = LDA(X, y, 2)
    X_new = np.dot(X, W)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
    plt.show()

    # 和sklearn的函数对比
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    X_new = lda.transform(X)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
    plt.show()


main()
