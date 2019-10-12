import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn as ml

# 查看范例
# ml.plots.plot_kmeans_algorithm()
# ml.plots.plot_kmeans_boundaries()

# 从scikit-learn工具包的datasets包来加载数据集：
from sklearn.datasets import make_blobs
# 生成模拟的二维数据
X, y = make_blobs(random_state=1)

# 构建聚类模型
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# 输出各个样本所属的群组
print("Cluster memberships:\n{}".format(kmeans.labels_))
#  print(kmeans.predict(X))
ml.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
ml.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2],
                    markers='^', markeredgewidth=2)


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# k=2：
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
assignments = kmeans.labels_
ml.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

# k=4：
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
assignments = kmeans.labels_
ml.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])

plt.show()
