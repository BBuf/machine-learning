# coding=utf-8
from copy import copy
from random import randint, seed, random
from time import time

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

# 创建Node类
class Node(object):
    # 初始化，存储预测值，左右节点，特征和分割点

    def __init__(self, score=None):
        self.score = score
        self.left = None
        self.right = None
        self.feature = None
        self.split = None


# 创建回归树类
class RegressionTree(object):

    # 初始化，存储根节点和树的高度
    def __init__(self):
        self.root = Node()
        self.height = 0

    # 计算分割点，MSE， 根据自变量X、因变量y、X元素中被取出的行号idx，
    # 列号feature以及分割点split，计算分割后的MSE。注意这里为了减少
    # 计算量，用到了方差公式：D(X)=E{[X-E(X)]^2}=E(X^2)-[E(X)]^2
    def get_split_mse(self, X, y, idx, feature, split):
        split_sum = [0, 0]
        split_cnt = [0, 0]
        split_sqr_sum = [0, 0]
        for i in idx:
            xi, yi = X[i][feature], y[i]
            if xi < split:
                split_cnt[0] += 1
                split_sum[0] += yi
                split_sqr_sum[0] += yi ** 2
            else:
                split_cnt[1] += 1
                split_sum[1] += yi
                split_sqr_sum[1] += yi ** 2
        split_avg = [split_sum[0] / split_cnt[0], split_sum[1] / split_cnt[1]]
        split_mse = [split_sqr_sum[0] - split_sum[0] * split_avg[0],
                     split_sqr_sum[1] - split_sum[1] * split_avg[1]]
        return sum(split_mse), split, split_avg

    # 计算最佳分割点，遍历特征某一列的所有的不重复的点，找出MSE最小的点
    # 作为最佳分割点。如果特征中没有不重复的元素则返回None。
    def choose_split_point(self, X, y, idx, feature):
        unique = set([X[i][feature] for i in idx])
        if (len(unique) == 1):
            return None
        unique.remove(min(unique))
        mse, split, split_avg = min((self.get_split_mse(X, y, idx, feature, split)
                                     for split in unique), key=lambda x: x[0])
        return mse, feature, split, split_avg

    # 选择最佳特征，遍历所有特征，计算最佳分割点对应的MSE，找出MSE最小
    # 的特征、对应的分割点，左右子节点对应的均值和行号。如果所有的特征都没有不重复元素则返回None
    def choose_feature(self, X, y, idx):
        m = len(X[0])
        split_rets = [x for x in map(lambda x: self.choose_split_point(X, y, idx, x),
                                     range(m)) if x is not None]
        if (split_rets == []):
            return None
        _, feature, split, split_avg = min(split_rets, key=lambda x: x[0])

        idx_split = [[], []]
        while idx:
            i = idx.pop()
            xi = X[i][feature]
            if xi < split:
                idx_split[0].append(i)
            else:
                idx_split[1].append(i)
        return feature, split, split_avg, idx_split

    # 规则转文字
    def expr2literal(self, expr):
        feature, op, split = expr
        op = ">=" if op == 1 else "<"
        return ("Feature%d %s %.4f" % (feature, op, split))

    # 获取规则，将回归树的所有规则都用文字表达出来，方便我们了解树的全貌。这里使用BFS。
    def get_rules(self):
        que = [[self.root, []]]
        self.rules = []
        while que:
            nd, exprs = que.pop(0)
            if not (nd.left or nd.right):
                literals = list(map(self.expr2literal, exprs))
                self.rules.append([literals, nd.score])
            if nd.left:
                rule_left = copy(exprs)
                rule_left.append([nd.feature, -1, nd.split])
                que.append([nd.left, rule_left])

            if nd.right:
                rule_right = copy(exprs)
                rule_right.append([nd.feature, 1, nd.split])
                que.append([nd.right, rule_right])

    # 训练模型，仍然使用队列+广度优先搜索，训练模型的过程中需要注意:
    # 1.控制树的最大深度max_depth
    # 2.控制分裂时最少的样本量min_samples_split
    # 3.叶子节点至少有两个不重复的y值
    # 4.至少有一个特征是没有重复值的
    def fit(self, X, y, max_depth=5, min_samples_split=2):
        self.root = Node()
        que = [[0, self.root, list(range(len(y)))]]
        while que:
            depth, nd, idx = que.pop(0)
            if depth > max_depth:
                depth -= 1
                break
            if len(idx) < min_samples_split or set(map(lambda i: y[i], idx)) == 1:
                continue
            feature_rets = self.choose_feature(X, y, idx)
            if feature_rets is None:
                continue
            nd.feature, nd.split, split_avg, idx_split = feature_rets
            nd.left = Node(split_avg[0])
            nd.right = Node(split_avg[1])
            que.append([depth + 1, nd.left, idx_split[0]])
            que.append([depth + 1, nd.right, idx_split[1]])

        self.height = depth
        self.get_rules()

    # 打印规则
    def print_reuls(self):
        for i, rule in enumerate(self.rules):
            literals, score = rule
            print("Rule %d: " % i, ' | '.join(
                literals) + ' => split_hat %.4f' % score)
    # 预测一个样本
    def _predict(self, row):
        nd = self.root
        while nd.left and nd.right:
            if row[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return nd.score

    # 预测多个样本
    def predict(self, X):
        return [self._predict(Xi) for Xi in X]


# 效果评估
if __name__ == '__main__':
    print('Testing the accuracy of RegressionTree...')
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=10)
    reg = RegressionTree()
    reg.fit(X=X_train, y=y_train, max_depth=4)
    reg.print_reuls()
    get_r2(reg, X_test, y_test)
