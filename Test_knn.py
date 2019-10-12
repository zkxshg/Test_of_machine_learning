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
# 将数据划分为训练集train和测试集test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 导入sklearn的kNN分类器
from sklearn.neighbors import KNeighborsClassifier
# 创建3NN模型，调整n_neighbors参数可以设置不同的k值
clf = KNeighborsClassifier(n_neighbors=3)
# 用划分得到的训练数据训练3NN模型
clf.fit(X_train, y_train)

# 使用3NN模型进行预测并输出预测结果
print("Test set predictions: {}".format(clf.predict(X_test)))
# 计算并输出预测准确率
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# 绘制出3NN模型的决策边界
ml.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, alpha=.4)
ml.discrete_scatter(X[:, 0], X[:, 1], y)
plt.show()
