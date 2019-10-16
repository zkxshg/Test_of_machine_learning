import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn as ml

# 查看kNN范例
# 1NN范例
ml.plots.plot_knn_classification(n_neighbors=1)
# 3NN范例
ml.plots.plot_knn_classification(n_neighbors=3)

# 导入划分训练集和测试集的工具包
from sklearn.model_selection import train_test_split
# 读入待分类数据，X为特征，y为目标类别
X, y = ml.datasets.make_forge()

# 导入sklearn的kNN分类器
from sklearn.neighbors import KNeighborsClassifier
# 创建空白图例
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# 绘制出k = 1,3,9时的不同分类模型及决策边界
for n_neighbors, ax in zip([1, 3, 9], axes):
    # 创建并训练kNN模型
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    # 绘制分类模型结果和决策边界
    ml.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    ml.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    # 赋予标题和坐标轴名
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
    axes[0].legend(loc=3)

plt.show()

