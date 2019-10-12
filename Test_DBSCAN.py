import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn as ml

# 查看DBSCAN范例
# ml.plots.plot_dbscan()

# 读入半月形分类数据集moons
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# 导入数据缩放工具包
from sklearn.preprocessing import StandardScaler
# 将数据缩放成平均值为0、方差为1，以方便进行密度聚类
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# 导入DBSCAN工具包
from sklearn.cluster import DBSCAN
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
# 绘制簇分配
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=ml.cm2, s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
