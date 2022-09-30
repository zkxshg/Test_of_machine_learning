import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn as ml
import nltk

# 读入数据
# 导入 sklearn 的 load_files 工具
from sklearn.datasets import load_files

# 加载训练数据集
# ！！！将括号内改为电影评论数据集的train文件夹所在位置！！！
reviews_train = load_files("C:/Users/zkx/Desktop/人工智能案例/人工智能案例-文本挖掘/aclImdb/train")
# 查看读入的数据
text_train, y_train = reviews_train.data, reviews_train.target
print("数据类型: {}".format(type(text_train)))
print("训练文本的文档数: {}".format(len(text_train)))
print("训练文本[1]:\n{}".format(text_train[1]))
print("正向/负向训练样本个数: {}".format(np.bincount(y_train)))


# 建立语料库
from sklearn.feature_extraction.text import CountVectorizer
# 实例化转换器并加入停用词表
vect = CountVectorizer(max_features=10000, min_df=5, max_df=.15, stop_words="english").fit(text_train)
# 创建语料库
X_train = vect.transform(text_train)

# 建立主题模型
# 导入 sklearn 中的 LDA 工具包
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(learning_method="batch", max_iter=25, random_state=0)
# 创建lda主题模型
document_topics = lda.fit_transform(X_train)

#将结果按主题排序并输出前十大主题及其10个代表词汇
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
ml.tools.print_topics(topics=range(10), feature_names=feature_names, sorting=sorting, topics_per_chunk=5, n_words=10)

# 找出主题0的评论，并输出前十条评论
# 修改 document_topics[:, 0] 中的 0 可以输出其他主题的评论，如 document_topics[:, 3] 即为主题3的评论
war = np.argsort(document_topics[:, 0])[::-1]
for i in war[:10]:
    print(b".".join(text_train[i].split(b".")[:2]) + b".\n")
plt.show()
