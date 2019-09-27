#coding=utf-8
from kdtree import KDTree
from max_heap import MaxHeap

from copy import copy
from random import randint, seed, random
from time import time
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

def load_house_data():
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

class KNeighborsBase(object):

    def __init__(self):
        self.k_neighbors = None
        self.tree = None

    def fit(self, X, y, k_neighbors=3):
        self.k_neighbors = k_neighbors
        self.tree = KDTree()
        self.tree.build_tree(X, y)

    # 1.获取kd_Tree
    # 2.建立大顶堆
    # 3.建立队列
    # 4.外层循环更新大顶堆
    # 5.内层循环遍历kd_Tree
    # 6.满足堆顶是第k近邻时退出循环

    def knn_search(self, Xi):
        tree = self.tree
        heap = MaxHeap(self.k_neighbors, lambda x: x.dist)
        # 搜索Xi时，从根节点到叶节点的路径
        nd = tree.search(Xi, tree.root)
        # 初始化队列
        que = [(tree.root, nd)]
        while que:
            # 计算Xi和根节点的距离
            nd_root, nd_cur = que.pop(0)
            nd_root.dist = tree.get_eu_dist(Xi, nd_root)
            heap.add(nd_root)
            while nd_cur is not nd_root:
                # 计算Xi和当前节点的距离
                nd_cur.dist = tree.get_eu_dist(Xi, nd_cur)
                # 更新最好的节点和距离
                heap.add(nd_cur)
                if nd_cur.brother and (not heap or heap.items[0].dist > tree.get_hyper_plane_dist(Xi, nd_cur.father)):
                    _nd = tree.search(Xi, nd_cur.brother)
                    que.append((nd_cur.brother, _nd))
                nd_cur = nd_cur.father

        return heap

    def _predict(self, Xi):
        return NotImplemented

    def predict(self, X):
        return [self._predict(Xi) for Xi in X]


class KNeighborsClassifier(KNeighborsBase):
    def __init__(self):
        KNeighborsBase.__init__(self)

    def _predict(self, Xi):
        heap = self.knn_search(Xi)
        n_pos = sum(nd.split[1] for nd in heap._items)
        return int(n_pos * 2 > self.k_neighbors)

class KNeighborsRegressor(KNeighborsBase):
    def __init__(self):
        KNeighborsBase.__init__(self)

    def _predict(self, Xi):
        heap = self.knn_search(Xi)
        return sum(nd.split[1] for nd in heap._items) / self.k_neighbors


@run_time
def main1():
    print("Tesing the performance of KNN classifier...")
    X, y = load_cancer()
    X = min_max_scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train, k_neighbors=21)
    model_evaluation(clf, X_test, y_test)

@run_time
def main2():
    print("Tesing the performance of KNN regressor...")
    X, y = load_house_data()
    X = min_max_scale(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=10)
    reg = KNeighborsRegressor()
    reg.fit(X=X_train, y=y_train, k_neighbors=3)
    get_r2(reg, X_test, y_test)