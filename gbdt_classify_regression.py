#coding=utf-8
from copy import copy
from random import randint, seed, random
from time import time
from regression import RegressionTree
from random import choice

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


class GradientBoostingBase(object):
    # 初始化，存储回归树、学习率、初始预测值和变换函数。
    # （注：回归不需要做变换，因此函数的返回值等于参数）
    def __init__(self):
        self.trees = None
        self.lr = None
        self.init_val = None
        self.fn = lambda x: x

    # 计算初始预测值，初始预测值即y的平均值。
    def get_init_val(self, y):
        return sum(y) / len(y)

    # 计算残差
    def get_residuals(self, y, y_hat):
        return [yi - self.fn(y_hat_i) for yi, y_hat_i in zip(y, y_hat)]

    # 找到例子属于哪个分类的叶子节点
    def match_node(self, row, tree):
        nd = tree.root
        while nd.left and nd.right:
            if row[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return nd

    # 得到回归树的所有的叶子节点
    def get_leaves(self, tree):
        nodes = []
        que = [tree.root]
        while que:
            node = que.pop(0)
            if node.left is None or node.right is None:
                nodes.append(node)
                continue
            left_node = node.left
            right_node = node.right
            que.append(left_node)
            que.append(right_node)
        return nodes

    # 将样本的索引划分为回归树的相应叶节点。
    # 返回一个字典，类似于:{node1: [1, 3, 5], node2: [2, 4, 6]...}，代表哪个节点对哪些样本进行了决策(分类)
    def divide_regions(self, tree, nodes, X):
        regions = {node: [] for node in nodes}
        for i, row in enumerate(X):
            node = self.match_node(row, tree)
            regions[node].append(i)
        return regions

    # 计算回归树的叶子节点值
    def get_score(self, idxs, y_hat, residuals):

        return None


    # 更新回归树的叶子节点值
    def update_score(self, tree, X, y_hat, residuals):
        nodes = self.get_leaves(tree)
        regions = self.divide_regions(tree, nodes, X)
        for node, idxs in regions.items():
            node.score = self.get_score(idxs, y_hat, residuals)
        tree.get_rules()

    # 训练模型的时候需要注意以下几点：
    # 1.控制树的最大深度max_depth；
    # 2.控制分裂时最少的样本量min_samples_split；
    # 3.训练每一棵回归树的时候要乘以一个学习率lr，防止模型过拟合；
    # 4.对样本进行抽样的时候要采用有放回的抽样方式。
    def fit(self, X, y, n_estimators, lr, max_depth, min_samples_split, subsample=None):
        self.init_val = self.get_init_val(y)
        n = len(y)
        y_hat = [self.init_val] * n
        residuals = self.get_residuals(y, y_hat)

        self.trees = []
        self.lr = lr
        for _ in range(n_estimators):
            idx = range(n)
            if subsample is not None:
                k = int(subsample * n)
                idx = choices(population=idx, k=k)
            X_sub = [X[i] for i in idx]
            residuals_sub = [residuals[i] for i in idx]
            y_hat_sub = [y_hat[i] for i in idx]
            tree = RegressionTree()
            tree.fit(X_sub, residuals_sub, max_depth, min_samples_split)
            # Update scores of tree leaf nodes
            self.update_score(tree, X_sub, y_hat_sub, residuals_sub)
            # Update y_hat
            # y_hat = [y_hat_i + lr * res_hat_i for y_hat_i, res_hat_i in zip(y_hat, tree.predict(X))]
            # Update residuals
            residuals = self.get_residuals(y, y_hat)
            self.trees.append(tree)

    # 对单个样本进行预测
    def _predict(self, Xi):
        ret = self.init_val + sum(self.lr * tree._predict(Xi) for tree in self.trees)
        return self.fn(ret)

    # 对多个样本进行预测
    def predict(self, X):
        #return [self._predict(Xi) for Xi in X]
        return None
