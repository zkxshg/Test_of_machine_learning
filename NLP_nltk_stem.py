import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn as ml
import nltk

# 定义待处理的原始词汇，可以随意对下方词汇做添加或修改
plurals = ['caresses', 'flies', 'dies', 'mules', 'denied', 'died', 'agreed', 'owned', 'humbled', 'sized',
           'meeting', 'stating', 'siezing', 'itemization',
           'sensational', 'traditional', 'reference', 'colonizer','plotted']

# 将nltk的Porter词干提取器实例化
stemmer = nltk.stem.PorterStemmer()
# 对上方词汇进行词干提取
singles = [stemmer.stem(plural) for plural in plurals]

# 输出提取前后词汇对比
print('未提取原始词汇  词干提取结果')
for i in range(len(plurals)): print('%s   %s '%(plurals[i], singles[i]))
