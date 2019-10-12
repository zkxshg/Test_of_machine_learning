import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn as ml

# 查看决策树范例
ml.plots.plot_animal_tree()
plt.show()

# 从scikit-learn工具包的datasets包来加载数据集：
from sklearn.datasets import load_breast_cancer
# 导入癌症诊断数据集
cancer = load_breast_cancer()

# 导入划分训练集和测试集的工具包
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# 创建决策树模型
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
# 输出模型概要和测试准确率
print(tree)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# 导入sklearn决策树可视化工具包
from sklearn.tree import export_graphviz
# 将绘制得到的决策树模型保存为dot格式图形文件
export_graphviz(tree, out_file="tree.dot", class_names=["malignant","benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)

# 通过graphviz工具包绘制生成的决策树模型
import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()

gra = graphviz.Source(dot_graph)
gra.view()
