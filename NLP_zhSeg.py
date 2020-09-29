# 安装并验证jieba
# pip install jieba
import jieba
import jieba.posseg

# 用样例验证是否安装成功
seg_list = jieba.cut("基于结巴的中文分词和词性标注", cut_all=True)
print(",".join(seg_list)) 

# short cut
string = '两岸的豆麦和河底的水草所发散出来的清香，夹杂在水气中扑面的吹来；月色便朦胧在这水气里。' \
         '淡黑的起伏的连山，仿佛是踊跃的铁的兽脊似的，都远远的向船尾跑去了，但我却还以为船慢。' \
         '他们换了四回手，渐望见依稀的赵庄，而且似乎听到歌吹了，还有几点火，料想便是戏台，但或者也许是渔火'

# 中文分词
seg = jieba.cut(string)
# 输出分词结果
l = []
for i in seg:
    l.append(i)
print(l)

# 中文词性标注
seg2 = jieba.posseg.cut(string)
# 输出词性标注结果
l2 = []
for i in seg2:
    l2.append((i.word, i.flag))
print(l2)

# pip install nltk
# pip install matplotlib
import nltk
import jieba


# 读入目标文本，注意文本是ANSI编码
raw=open(u'神雕侠侣.txt',encoding='gb18030', errors='ignore').read()

# 使用 jieba 的分词词库结合 nltk 进行分词
text=nltk.text.Text(jieba.lcut(raw))

# 找出“金轮法王”在全文出现了多少次
print(text.concordance(u'金轮法王'))

import matplotlib.pyplot as plt

# 查看 李莫愁 和 郭襄 两个词在全文的分布
text.dispersion_plot([u'李莫愁', u'郭襄'])
plt.show()
