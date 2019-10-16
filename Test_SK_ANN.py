import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn as ml

# 读入半月形分类数据集moons
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

# 导入划分训练集和测试集的工具包
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 创建多层感知机模型
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)

# 绘制出多层感知机的分类边界
ml.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
ml.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

plt.show()
