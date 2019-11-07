import numpy as np
# 零均值化，即中心化，是数据的预处理方法
def zero_centered(data):
    matrix_mean = np.mean(data, axis=0)
    return data - matrix_mean

def pca_eig(data, n):
    new_data = zero_centered(data)
    conv_mat = np.dot(new_data.T, new_data) #也可以用np.cov()方法
    eig_values, eig_vectors = np.linalg.eig(np.mat(conv_mat))
    # 求特征值和特征向量，特征向量是列向量
    value_indices = np.argsort(eig_values) #将特征值从小到大排序
    n_vectors = eig_vectors[:, value_indices[-1: -(n+1): -1]]
    # 最大的n个特征值对应的特征向量
    return new_data * n_vectors #返回低维特征空间的数据

def pca_svd(data, n):
    new_data = zero_centered(data)
    cov_mat = np.dot(new_data.T, new_data)
    U, s, V = np.linalg.svd(cov_mat) #将协方差矩阵奇异值分解
    pc = np.dot(new_data, U) #返回矩阵的第一个列向量即是降维后的结果
    return pc[:, 0]

def unit_test():
    data = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], 
                     [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])
    result_eig = pca_eig(data, 1)
    # 使用常规的特征值分解法，将二维数据降到一维
    print(result_eig)
    result_svd = pca_svd(data, 1)
    # 使用奇异值分解法将协方差矩阵分解，得到降维结果
    print(result_svd)
unit_test()
