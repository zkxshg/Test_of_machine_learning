import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn as ml

# 查看凝聚聚类范例
# ml.plots.plot_agglomerative_algorithm()

# 从scikit-learn工具包的datasets包来加载数据集：
from sklearn.datasets import make_blobs
# 生成模拟的二维数据
X, y = make_blobs(random_state=1)

# 读入凝聚聚类工具
from sklearn.cluster import AgglomerativeClustering
# 创建凝聚聚类模型：k=3
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)
ml.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

plt.show()
