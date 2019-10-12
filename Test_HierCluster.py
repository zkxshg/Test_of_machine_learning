import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn as ml


# 查看凝聚聚类过程范例
# ml.plots.plot_agglomerative()

# 从scikit-learn工具包的datasets包来加载数据集：
from sklearn.datasets import make_blobs
# 生成模拟的二维数据
X, y = make_blobs(random_state=0, n_samples=12)

# 从SciPy中导入dendrogram函数和ward聚类函数
from scipy.cluster.hierarchy import dendrogram, ward

# 将ward聚类应用于数据数组X
# ward函数返回一个数组，指定执行凝聚聚类时跨越的距离
linkage_array = ward(X)

# 为包含簇之间距离的linkage_array绘制树状图
dendrogram(linkage_array)

# 在树中标记划分成两个簇或三个簇的位置
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')
ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")

plt.show()
