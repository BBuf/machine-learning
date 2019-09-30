#coding=utf-8

from collections import defaultdict
from random import random
from itertools import product, chain

class Matrix(object):
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]))

    def row(self, row_no):
        return Matrix([self.data[row_no]])

    def col(self, col_no):
        m = self.shape[0]
        return Matrix([[self.data[i][col_no] for i in range(m)]])

    @property
    def is_squre(self):
        return self.shape[0] == self.shape[1]

    @property
    def transpose(self):
        data = list(map(list, zip(*self.data)))
        return data

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



class ALS(object):
