from jieba import analyse
import jieba

# 引入TextRank关键词抽取接口
textrank = analyse.textrank

# 读入原始文本
article = open(u'神雕侠侣_ANSI.txt', encoding='gb18030', errors='ignore').read()
# 对原文进行分词
words = jieba.cut(article, cut_all=False)

# 记录所有词的出现频率
word_freq = {}
for word in words:
    if word in word_freq:
        word_freq[word] += 1
    else:
        word_freq[word] = 1

# 创建词-词频对
freq_word = []
for word, freq in word_freq.items():
    freq_word.append((word, freq))
freq_word.sort(key=lambda x: x[1], reverse=True)  # 反序排列，根据第二个参数

# 输出的前max_number个高频词，修改max_number可以修改输出的高频词个数
max_number = 20
for word, freq in freq_word[: max_number]:
    print(word, freq)
