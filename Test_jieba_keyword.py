import nltk
import jieba
from jieba import analyse
import matplotlib.pyplot as plt

# 引入TextRank关键词抽取接口
textrank = analyse.textrank

# 读入目标文本，注意文本是ANSI编码
raw=open(u'神雕侠侣_ANSI.txt',encoding='gb18030', errors='ignore').read()

# 抽取目标文章的关键字
keywords = textrank(raw)

# 输出抽取的关键字
for keyword in keywords:
    print(keyword + "/")
