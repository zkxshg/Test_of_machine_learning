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
vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
print("训练数据语料库信息:\n{}".format(repr(X_train)))
# 查看语料库信息
feature_names = vect.get_feature_names()
print("================================语料库=======================================")
print("词汇总量: {}".format(len(feature_names)))
print("前20个词汇:\n{}".format(feature_names[:20]))
print("第2000个到第2030个词汇:\n{}".format(feature_names[2000:2030]))
print("第2000*x个词汇:\n{}".format(feature_names[::2000]))
print('Type of feature')
print(type(feature_names))

# 剔除停用词
# 导入 sklearn 中的英文停用词表
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# 查看英文停用词表信息
print("================================停用词=======================================")
print("停用词总数: {}".format(len(ENGLISH_STOP_WORDS)))
print("第10*x个停用词:\n{}".format(list(ENGLISH_STOP_WORDS)[::10]))

# 剔除语料库中的停用词
vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train)
X_train = vect.transform(text_train)
print("================================剔除停用词后=======================================")
feature_names = vect.get_feature_names()
print("剔除停用词后的词汇总量:\n{}".format(len(feature_names)))
print("前20个词汇:\n{}".format(feature_names[:20]))
print("第2000个到第2030个词汇:\n{}".format(feature_names[2000:2030]))
print("第2000*x个词汇:\n{}".format(feature_names[::2000]))
