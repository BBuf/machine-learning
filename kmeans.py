#coding=utf-8
from collections import Counter
from copy import deepcopy
from random import random, randint

class KMeans(object):
    # k 簇的个数
    # n_features 特征的个数
    # clister_centers 聚类中心
    # distance_fn 距离计算函数
    # cluster_samples_cnt 每个簇里面的样本数
    def __init__(self):
        self.k = None
        self.n_features = None
        self.cluster_centers = None
        self.distance_fn = None
        self.cluster_samples_cnt = None

    # 二分，查找有序列表里面大于目标值的第一个值
    def bin_search(self, target, nums):
        low = 0
        high = len(nums) - 1
        assert nums[low] <= target < nums[high], "Cannot find target!"
        while 1:
            mid = (low + high) // 2
            if mid == 0 or target >= nums[mid]:
                low = mid + 1
            elif target < nums[mid - 1]:
                high = mid - 1
            else:
                break
        return mid

    # 初始化聚类中心
    def init_cluster_centers(self, X, k, n_features, distance_fn):
        n = len(X)
        centers = [X[randint(0, n-1)]]
        for _ in range(k-1):
            center_pre = centers[-1]
            