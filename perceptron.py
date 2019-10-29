#coding=utf-8
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
# 行列数据标注
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
# print (df.label.value_counts())
print(df.head(10))

# 数据可视化
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], c='red', label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], c='blue', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
#plt.show()

# 选择特征和标签
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y]) #将label中的0标签替换为-1

# 开始实现感知机算法

class Model:
    # 初始化
    def __init__(self):
        # 初始化权重
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        # 初始化偏执
        self.b = 0
        # 学习率
        self.l_rate = 0.1

    # 定义符号函数sign
    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    # 随机梯度下降法
    def fit(self, X_train, y_train):
        is_wrong = False
        while not is_wrong:
            wrong_cnt = 0
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]
                if (y * self.sign(X, self.w, self.b) <= 0):
                    # 更新权重
                    self.w = self.w + self.l_rate * np.dot(y, X)
                    # 更新步长
                    self.b = self.b + self.l_rate * y
                    wrong_cnt += 1
            if(wrong_cnt == 0):
                is_wrong  = True

        return 'Perceptron Model!'

    def score(self):
        pass


# 开始调用感知机模型
perceptron = Model()
perceptron.fit(X, y)
# 可视化超平面
x_points = np.linspace(4, 7, 10)
# 误分类点到超平面的距离
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
plt.plot(x_points, y_)
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], c='red', label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], c='blue', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
