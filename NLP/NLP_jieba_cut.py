import jieba
import jieba.posseg

# 创建一个简单的文段
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
