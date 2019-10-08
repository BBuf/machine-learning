#coding=utf-8
from math import log2
from copy import copy
from time import time
from random import random
from random import randint, seed, random
import numpy as np

def list_split(X, idxs, feature, split):
    ret = [[], []]
    while idxs:
        if X[idxs[0]][feature] < split:
            ret[0].append(idxs.pop(0))
        else:
            ret[1].append(idxs.pop(0))
    return ret

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

# 定义决策树的节点
class Node(object):
    def __init__(self, prob=None):
        self.prob = prob
        self.left = None
        self.right = None
        self.feature = None
        self.split = None


class DecisionTree(object):
    # 本决策树只支持使用ID3的二分类
    # root代表根节点，depth代表决策树的深度
    def __init__(self):
        self.root = Node()
        self.depth = 1
        self._rules = None

    # 对特定特征feature和分裂点x
    def get_split_effect(self, X, y, idx, feature, split):
        n = len(idx)
        pos_cnt = [0, 0]
        cnt = [0, 0]
        for i in idx:
            xi, yi = X[i][feature], y[i]
            if xi < split:
                cnt[0] += 1
                pos_cnt[0] += yi
            else:
                cnt[1] += 1
                pos_cnt[1] += y[i]
        # 计算分裂影响
        prob = [pos_cnt[0] / cnt[0], pos_cnt[1] / cnt[1]]
        rate = [cnt[0] / n, cnt[1] / n]
        return prob, rate

    # 计算熵
    def get_entropy(self, p):
        if p == 1 or p == 0:
            return 0
        else:
            q = 1 - p
            return -(p * log2(p) + q * log2(q))

    # 计算信息熵
    def get_info(self, y, idx):
        p = sum(y[i] for i in idx) / len(idx)
        return self.get_entropy(p)

    # 计算条件熵
    def get_cond_info(self, prob, rate):
        info_left = self.get_entropy(prob[0])
        info_right = self.get_entropy(prob[1])
        return rate[0] * info_left + rate[1] * info_right
    # 寻找feature特征下的最好的分裂点
    def choose_split(self, X, y, idxs, feature):
        unique = set([X[i][feature] for i in idxs])
        if len(unique) == 1:
            return None
        unique.remove(min(unique))
        def f(split):
            info = self.get_info(y, idxs)
            prob, rate = self.get_split_effect(X, y, idxs, feature, split)
            cond_info = self.get_cond_info(prob, rate)
            # 信息增益
            gain = info - cond_info
            return gain, split, prob
        # 得到用于最大信息增益的分裂点
        gain, split, prob = max((f(split) for split in unique), key=lambda x: x[0])

        return gain, feature, split, prob

    # 寻找具有最大信息增益的特征
    def choose_feature(self, X, y, idxs):
        m = len(X[0])
        split_rets = map(lambda j: self.choose_split(X, y, idxs, j), range(m))
        split_rets = filter(lambda x: x is not None, split_rets)
        return max(split_rets, default=None, key=lambda x: x[0])

    def expr2literal(self, expr):
        feature, op, split = expr
        op = ">=" if op == 1 else "<"
        return "Feature%d %s %.4f" % (feature, op, split)

    # 获取决策树所有叶子节点的规则
    def get_rules(self):
        que = [[self.root, []]]
        self._rules = []
        while que:
            nd, exprs = que.pop(0)
            if not(nd.left or nd.right):
                literals = list(map(self.expr2literal, exprs))
                self._rules.append([literals, nd.prob])
            if nd.left:
                rule_left = copy(exprs)
                rule_left.append([nd.feature, -1, nd.split])
                que.append([nd.left, rule_left])
            if nd.right:
                rule_right = copy(exprs)
                rule_right.append([nd.feature, 1, nd.split])
                que.append([nd.right, rule_right])

    # 训练数据
    def fit(self, X, y, max_depth=4, min_samples_split=2):
        idxs = list(range(len(y)))
        que = [(self.depth+1, self.root, idxs)]
        while que:
            depth, nd, idxs = que.pop(0)
            if depth > max_depth:
                depth -= 1
                break
            if len(idxs) < min_samples_split or nd.prob == 1 or nd.prob == 0:
                continue
            split_ret = self.choose_feature(X, y, idxs)
            if split_ret is None:
                continue
            _, feature, split, prob = split_ret
            nd.feature = feature
            nd.split = split
            nd.left = Node(prob[0])
            nd.right = Node(prob[1])
            idxs_split = list_split(X, idxs, feature, split)
            que.append((depth + 1, nd.left, idxs_split[0]))
            que.append((depth + 1, nd.right, idxs_split[1]))
        self.depth = depth
        self.get_rules()

    def print_rules(self):
        for i, rule in enumerate(self._rules):
            literals, prob = rule
            print("Rule %d: " % i, ' | '.join(literals) + ' => y_hat %.4f' % prob)
            print()
    def _predict(self, Xi):
        nd = self.root
        while nd.left and nd.right:
            if Xi[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return nd.prob

    def predict(self, X, threshold=0.5):
        return [int(self._predict(Xi) >= threshold) for Xi in X]


@run_time
# 效果评估
def main():
    print("Tesing the performance of DecisionTree...")
    data, label = load_data()
    data_train, data_test, label_train, label_test = train_test_split(data, label, random_state=100)
    clf = DecisionTree()
    clf.fit(data_train, label_train)
    clf.print_rules()
    y_hat = clf.predict(data_test)
    acc = get_acc(label_test, y_hat)
    print("Accuracy is %.3f" % acc)
