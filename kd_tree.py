#coding=utf-8
from time import time
from copy import copy
from random import randint

# 产生一个[low,high)区间的随机数组
def gen_data(low, high, n_rows, n_cols=None):
    if n_cols is None:
        ret = [randint(low, high) for _ in range(n_rows)]
    else:
        ret = [[randint(low, high) for _ in range(n_cols)]
               for _ in range(n_rows)]
    return ret

def get_euclidean_distance(arr1, arr2):
    return sum((x1 - x2) ** 2 for x1, x2 in zip(arr1, arr2))

class Node(object):

    def __init__(self):
        self.father = None
        self.left = None
        self.right = None
        self.feature = None
        self.split = None

    def __str__(self):
        return ("feature: %s, split: %s" % (str(self.feature), str(self.split)))

    @property
    def brother(self):
        if not self.father:
            ret = None
        else:
            if self.father.left is self:
                ret = self.father.right
            else:
                ret = self.father.left
        return ret

class KDTree(object):

    def __init__(self):
        # root代表KD-Tree的根节点
        self.root = Node()

    def __str__(self):
        # 展示KD-Tree每个节点的关系
        ret = []
        i = 0
        que = [(self.root, -1)]
        while que:
            nd, idx_father = que.pop(0)
            ret.append("%d -> %d: %s" % (idx_father, i, str(nd)))
            if nd.left:
                que.append((nd.left, i))
            if nd.right:
                que.append((nd.right, i))
            i += 1
        return "\n".join(ret)

    def get_median_idx(self, X, idxs, feature):
        # 计算一列数据的中位数
        n = len(idxs)
        # 忽略n为奇数或者偶数
        k = n // 2
        # 以元组形式获取列j的所有索引和元素
        col = map(lambda i: (i, X[i][feature]), idxs)
        # 根据元素的值对元组进行排序，并获取相应的索引
        sorted_idxs = map(lambda x: x[0], sorted(col, key=lambda x:x[1]))
        # 搜索中值
        median_idx = list(sorted_idxs)[k]
        return median_idx

    def get_variance(self, X, idxs, feature):
        # 计算一列数据的方差
        n = len(idxs)
        col_sum = col_sum_sqr = 0
        for idx in idxs:
            xi = X[idx][feature]
            col_sum += xi
            col_sum_sqr += xi ** 2
        # D(X) = E{[X-E(X)]^2} = E(X^2)-[E(X)]^2
        return col_sum_sqr / n - (col_sum / n) ** 2
    # 取方差最大的特征作为分割点特征
    def choose_feature(self, X, idxs):
        m = len(X[0])
        variances = map(lambda j: (j, self.get_variance(X, idxs, j)), range(m))
        return max(variances, key=lambda x: x[1])[0]

    def split_feature(self, X, idxs, feature, median_idx):
        idxs_split = [[], []]
        split_val = X[median_idx][feature]
        for idx in idxs:
            if(idx == median_idx):
                continue
            xi = X[idx][feature]
            if xi < split_val:
                idxs_split[0].append(idx)
            else:
                idxs_split[1].append(idx)
        return idxs_split

    # 使用广度优先搜索的方式建立KD Tree，注意要对X进行归一化。
    def build_tree(self, X, y):
        nd = self.root
        idxs = range(len(X))
        que = [(nd, idxs)]
        while que:
            nd, idxs = que.pop(0)
            n = len(idxs)
            if n == 1:
                nd.split = (X[idxs[0]], y[idxs[0]])
                continue
            feature = self.choose_feature(X, idxs)
            median_idx = self.get_median_idx(X, idxs, feature)
            idxs_left, idxs_right = self.split_feature(X, idxs, feature, median_idx)
            nd.feature = feature
            nd.split = (X[median_idx], y[median_idx])
            if idxs_left != []:
                nd.left = Node()
                nd.left.father = nd
                que.append((nd.left, idxs_left))
            if idxs_right != []:
                nd.right = Node()
                nd.right.father = nd
                que.append((nd.right, idxs_right))

    # 比较目标元素与当前结点的当前feature，访问对应的子节点。
    # 反复执行上述过程，直到到达叶子节点。
    def search(self, Xi, nd):
        while nd.left or nd.right:
            if nd.left is None:
                nd = nd.right
            elif nd.right is None:
                nd = nd.left
            else:
                if Xi[nd.feature] < nd.split[0][nd.feature]:
                    nd = nd.left
                else:
                    nd = nd.right
        return nd

    #计算目标元素与某个节点的欧氏距离，注意get_euclidean_distance
    # 这个函数没有进行开根号的操作，所以求出来的是欧氏距离的平方。
    def get_eu_dist(self, Xi, nd):
        X0 = nd.split[0]
        return get_euclidean_distance(Xi, X0)

    # 计算目标元素与某个节点所在超平面的欧氏距离，为了跟上面函数保持一致，要加上平方。
    def get_hyper_plane_dist(self, Xi, nd):
        j = nd.feature
        X0 = nd.split[0]
        return (Xi[j] - X0[j]) ** 2

    # 搜索KD-Tree中与目标元素距离最近的节点，使用广度优先搜索来实现。
    def nearest_neighbour_search(self, Xi):
        dist_best = float("inf")
        nd_best = self.search(Xi, self.root)
        que = [(self.root, nd_best)]
        while que:
            nd_root, nd_cur = que.pop(0)
            while 1:
                dist = self.get_eu_dist(Xi, nd_cur)
                if dist < dist_best:
                    dist_best = dist
                    nd_best = nd_cur
                if nd_cur is not nd_root:
                    nd_bro = nd_cur.brother
                    if nd_bro is not None:
                        dist_hyper = self.get_hyper_plane_dist(Xi, nd_cur.father)
                        if dist > dist_hyper:
                            _nd_best = self.search(Xi, nd_bro)
                            que.append((nd_bro, _nd_best))
                    nd_cur = nd_cur.father
                else:
                    break
        return nd_best


# 暴力搜索
def exhausted_search(X, Xi):
    dist_best = float('inf')
    row_best = None
    for row in X:
        dist = get_euclidean_distance(Xi, row)
        if dist < dist_best:
            dist_best = dist
            row_best = row
    return row_best


def main():
    print("Testing KD Tree...")
    test_times = 100
    run_time_1 = run_time_2 = 0
    for _ in range(test_times):
        low = 0
        high = 100
        n_rows = 1000
        n_cols = 2
        X = gen_data(low, high, n_rows, n_cols)
        y = gen_data(low, high, n_rows)
        Xi = gen_data(low, high, n_cols)

        tree = KDTree()
        tree.build_tree(X, y)

        start = time()
        nd = tree.nearest_neighbour_search(Xi)
        run_time_1 += time() - start
        ret1 = get_euclidean_distance(Xi, nd.split[0])

        start = time()
        row = exhausted_search(X, Xi)
        run_time_2 += time() - start
        ret2 = get_euclidean_distance(Xi, row)

        assert ret1 == ret2, "target:%s\nrestult1:%s\nrestult2:%s\ntree:\n%s" \
            % (str(Xi), str(nd), str(row), str(tree))
    print("%d tests passed!" % test_times)
    print("KD Tree Search %.2f s" % run_time_1)
    print("Exhausted search %.2f s" % run_time_2)


main()