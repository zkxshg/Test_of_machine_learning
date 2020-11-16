import nltk
import jieba
import matplotlib.pyplot as plt

# 读入目标文本，注意文本是ANSI编码
raw=open(u'神雕侠侣_ANSI.txt',encoding='gb18030', errors='ignore').read()

# 使用 jieba 的分词词库结合 nltk 进行分词
text=nltk.text.Text(jieba.lcut(raw))

# 找出“金轮法王”在全文出现了多少次
print(text.concordance(u'金轮法王'))

# 查看 李莫愁 和 郭襄 两个词在全文的分布
text.dispersion_plot([u'李莫愁', u'郭襄'])

plt.show()
