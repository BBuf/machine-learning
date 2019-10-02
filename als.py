#coding=utf-8

from collections import defaultdict
from random import random
from itertools import product, chain
from time import time

def load_movie_ratings():

    f = open("boston/movie_ratings.csv")
    lines = iter(f)
    col_names = ", ".join(next(lines)[:-1].split(",")[:-1])
    print("The column names are: %s." % col_names)
    data = [[float(x) if i == 2 else int(x)
             for i, x in enumerate(line[:-1].split(",")[:-1])]
            for line in lines]
    f.close()
    return data

class Matrix(object):
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]))

    def row(self, row_no):
        return Matrix([self.data[row_no]])

    def col(self, col_no):
        m = self.shape[0]
        return Matrix([[self.data[i][col_no]] for i in range(m)])

    @property
    def is_square(self):
        return self.shape[0] == self.shape[1]

    @property
    def transpose(self):
        data = list(map(list, zip(*self.data)))
        return Matrix(data)

    # 生成一个长度为n的单位阵
    def _eye(self, n):
        return [[0 if i != j else 1 for j in range(n)] for i in range(n)]

    @property
    def eye(self):
        assert self.is_squre, "The matrix has to be squre"
        data = self._eye(self.shape[0])
        return Matrix(data)

    # 高斯消元
    def gaussian_elimination(self, aug_matrix):
        n = len(aug_matrix)
        m = len(aug_matrix[0])

        # From top to bottom.
        for col_idx in range(n):
            # Check if element on the diagonal is zero.
            if aug_matrix[col_idx][col_idx] == 0:
                row_idx = col_idx
                # Find a row whose element has same column index with
                # the element on the diagonal is not zero.
                while row_idx < n and aug_matrix[row_idx][col_idx] == 0:
                    row_idx += 1
                # Add this row to the row of the element on the diagonal.
                for i in range(col_idx, m):
                    aug_matrix[col_idx][i] += aug_matrix[row_idx][i]

            # Elimiate the non-zero element.
            for i in range(col_idx + 1, n):
                # Skip the zero element.
                if aug_matrix[i][col_idx] == 0:
                    continue
                # Elimiate the non-zero element.
                k = aug_matrix[i][col_idx] / aug_matrix[col_idx][col_idx]
                for j in range(col_idx, m):
                    aug_matrix[i][j] -= k * aug_matrix[col_idx][j]

        # From bottom to top.
        for col_idx in range(n - 1, -1, -1):
            # Elimiate the non-zero element.
            for i in range(col_idx):
                # Skip the zero element.
                if aug_matrix[i][col_idx] == 0:
                    continue
                # Elimiate the non-zero element.
                k = aug_matrix[i][col_idx] / aug_matrix[col_idx][col_idx]
                for j in chain(range(i, col_idx + 1), range(n, m)):
                    aug_matrix[i][j] -= k * aug_matrix[col_idx][j]

        # Iterate the element on the diagonal.
        for i in range(n):
            k = 1 / aug_matrix[i][i]
            aug_matrix[i][i] *= k
            for j in range(n, m):
                aug_matrix[i][j] *= k

        return aug_matrix

    # 矩阵求逆
    def _inverse(self, data):
        n = len(data)
        unit_matrix = self._eye(n)
        aug_matrix = [a + b for a, b in zip(self.data, unit_matrix)]
        ret = self.gaussian_elimination(aug_matrix)

        return list(map(lambda x: x[n:], ret))

    # 矩阵求逆，原理：https://baike.baidu.com/item/%E9%AB%98%E6%96%AF%E6%B6%88%E5%85%83%E6%B3%95/619561?fr=aladdin
    @property
    def inverse(self):
        assert self.is_square, "The matrix has to be square!"
        data = self._inverse(self.data)

        return Matrix(data)

    def row_mul(self, row_A, row_B):
        return sum(x[0] * x[1] for x in zip(row_A, row_B))

    def _mat_mul(self, row_A, B):
        row_pairs = product([row_A], B.transpose.data)
        return [self.row_mul(*row_pair) for row_pair in row_pairs]

    def mat_mul(self, B):
        assert self.shape[1] == B.shape[0], "A's column count does not match B's row count!"
        return Matrix([self._mat_mul(row_A, B) for row_A in self.data])

    def _mean(self, data):
        m = len(data)
        n = len(data[0])
        ret = [0 for _ in range(n)]
        for row in data:
            for j in range(n):
                ret[j] += row[j] / m
        return ret
    def mean(self, data):
        return Matrix(self._mean(self.data))

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


class ALS(object):
    # 初始化，存储用户ID、物品ID、用户ID与用户矩阵列号的对应关系、物品ID
    # 与物品矩阵列号的对应关系、用户已经看过哪些物品、评分矩阵的Shape以及RMSE
    def __init__(self):
        self.user_ids = None
        self.item_ids = None
        self.user_ids_dict = None
        self.item_ids_dict = None
        self.user_matrix = None
        self.item_matrix = None
        self.user_items = None
        self.shape = None
        self.rmse = None

    # 对训练数据进行处理，得到用户ID、物品ID、用户ID与用户矩阵列号的对应关系、物
    # 品ID与物品矩阵列号的对应关系、评分矩阵的Shape、评分矩阵及评分矩阵的转置。
    def process_data(self, X):
        self.user_ids = tuple((set(map(lambda x: x[0], X))))
        self.user_ids_dict = dict(map(lambda x: x[::-1], enumerate(self.user_ids)))
        self.item_ids = tuple((set(map(lambda x: x[1], X))))
        self.item_ids_dict = dict(map(lambda x: x[::-1], enumerate(self.item_ids)))
        self.shape = (len(self.user_ids), len(self.item_ids))
        ratings = defaultdict(lambda : defaultdict(int))
        ratings_T = defaultdict(lambda : defaultdict(int))
        for row in X:
            user_id, item_id, rating = row
            ratings[user_id][item_id] = rating
            ratings_T[item_id][user_id] = rating
        err_msg = "Length of user_ids %d and ratings %d not match!" % (
            len(self.user_ids), len(ratings))
        assert len(self.user_ids) == len(ratings), err_msg
        err_msg = "Length of item_ids %d and ratings_T %d not match!" % (
            len(self.item_ids), len(ratings_T))
        assert len(self.item_ids) == len(ratings_T), err_msg
        return ratings, ratings_T

    # 用户矩阵乘以评分矩阵，实现稠密矩阵与稀疏矩阵的矩阵乘法，得到用户矩阵与评分矩阵的乘积。
    def users_mul_ratings(self, users, ratings_T):
        def f(users_row, item_id):
            user_ids = iter(ratings_T[item_id].keys())
            scores = iter(ratings_T[item_id].values())
            col_nos = map(lambda x: self.user_ids_dict[x], user_ids)
            _users_row = map(lambda x: users_row[x], col_nos)
            return sum(a * b for a, b in zip(_users_row, scores))

        ret = [[f(users_row, item_id) for item_id in self.item_ids]
               for users_row in users.data]
        return Matrix(ret)

    # 物品矩阵乘以评分矩阵，实现稠密矩阵与稀疏矩阵的矩阵乘法，得到物品矩阵与评分矩阵的乘积。
    def items_mul_ratings(self, items, ratings):
        def f(items_row, user_id):
            item_ids = iter(ratings[user_id].keys())
            scores = iter(ratings[user_id].values())
            col_nos = map(lambda x: self.item_ids_dict[x], item_ids)
            _items_row = map(lambda x: items_row[x], col_nos)
            return sum(a * b for a, b in zip(_items_row, scores))

        ret = [[f(items_row, user_id) for user_id in self.user_ids]
               for items_row in items.data]
        return Matrix(ret)

    # 生成随机矩阵
    def gen_random_matrix(self, n_rows, n_colums):
        data = [[random() for _ in range(n_colums)] for _ in range(n_rows)]
        return Matrix(data)

    # 计算RMSE
    def get_rmse(self, ratings):
        m, n = self.shape
        mse = 0.0
        n_elements = sum(map(len, ratings.values()))
        for i in range(m):
            for j in range(n):
                user_id = self.user_ids[i]
                item_id = self.item_ids[j]
                rating = ratings[user_id][item_id]
                if rating > 0:
                    user_row = self.user_matrix.col(i).transpose
                    item_col = self.item_matrix.col(j)
                    rating_hat = user_row.mat_mul(item_col).data[0][0]
                    square_error = (rating - rating_hat) ** 2
                    mse += square_error / n_elements
        return mse ** 0.5

    # 训练模型
    # 1.数据预处理
    # 2.变量k合法性检查
    # 3.生成随机矩阵U
    # 4.交替计算矩阵U和矩阵I，并打印RMSE信息，直到迭代次数达到max_iter
    # 5.保存最终的RMSE
    def fit(self, X, k, max_iter=10):
        ratings, ratings_T = self.process_data(X)
        self.user_items = {k: set(v.keys()) for k,v in ratings.items()}
        m, n = self.shape
        error_msg = "Parameter k must be less than the rank of original matrix"
        assert k < min(m, n), error_msg
        self.user_matrix = self.gen_random_matrix(k, m)
        for i in range(max_iter):
            if i % 2:
                items = self.item_matrix
                self.user_matrix = self.items_mul_ratings(
                    items.mat_mul(items.transpose).inverse.mat_mul(items),
                    ratings
                )
            else:
                users = self.user_matrix
                self.item_matrix = self.users_mul_ratings(
                    users.mat_mul(users.transpose).inverse.mat_mul(users),
                    ratings_T
                )
            rmse = self.get_rmse(ratings)
            print("Iterations: %d, RMSE: %.6f" % (i + 1, rmse))
        self.rmse = rmse
    # 预测一个用户
    def _predict(self, user_id, n_items):
        users_col =  self.user_matrix.col(self.user_ids_dict[user_id])
        users_col = users_col.transpose

        items_col = enumerate(users_col.mat_mul(self.item_matrix).data[0])
        items_scores = map(lambda x: (self.item_ids[x[0]], x[1]), items_col)
        viewed_items = self.user_items[user_id]
        items_scores = filter(lambda x: x[0] not in viewed_items, items_scores)

        return sorted(items_scores, key=lambda x: x[1], reverse=True)[:n_items]

    # 预测多个用户
    def predict(self, user_ids, n_items=10):
        return [self._predict(user_id, n_items) for user_id in user_ids]

def format_prediction(item_id, score):
    return "item_id:%d score:%.2f" % (item_id, score)

@run_time
def main():
    print("Tesing the accuracy of ALS...")

    X = load_movie_ratings()

    model = ALS()
    model.fit(X, k=3, max_iter=5)
    print("Showing the predictions of users...")

    user_ids = range(1, 5)
    predictions = model.predict(user_ids, n_items=2)
    for user_id, prediction in zip(user_ids, predictions):
        _prediction = [format_prediction(item_id, score)
                       for item_id, score in prediction]
        print("User id:%d recommedation: %s" % (user_id, _prediction))


