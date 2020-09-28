# -*- coding: utf-8 -*-
# 2020-07-25
import pycorrector
import os
import pandas as pd
import numpy as np
import re
import nltk
import jieba
from jieba import analyse
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import corpora, models, similarities
# spacy for lemmatization
import spacy
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def correct(sentence, cor_path):
    # 是否开启错字纠错
    pycorrector.enable_char_error(enable=False)
    pycorrector.set_custom_confusion_dict(path=confusion_table)
    with open(cor_path, 'w+', encoding='utf-8') as f3:
        f3.write("")

    # ===== 打开纠正结果记录 =====
    with open(correct_result, 'a+', encoding='utf-8') as f2:
        f2.write("2.错别字纠正：\n")

    # 逐句纠错
    ifError = False
    for se in sentence:
        corrected_sent, detail = pycorrector.correct(se)
        if detail:
            ifError = True
            print("错句为：", se)
            print("纠正后结果为：", corrected_sent, "，纠错点为：", detail)
            with open(cor_path, 'a+', encoding='utf-8') as f2:
                # f2.write(corrected_sent + '\n')  # 加\n换行显示
                f2.write(corrected_sent)
            with open(correct_result, 'a+', encoding='utf-8') as f2:
                f2.write("错句为：" + se)
                f2.write("纠正后结果为：" + corrected_sent + "\n")
        else:
            with open(cor_path, 'a+', encoding='utf-8') as f2:
                # f2.write(se + '\n')  # 加\n换行显示
                f2.write(corrected_sent)
                
    # ===== 无错别字 =====
    if not ifError:
        with open(correct_result, 'a+', encoding='utf-8') as f2:
            f2.write("恭喜你，文章无错别字！" + "\n")
            f2.write("\n")
    with open(correct_result, 'a+', encoding='utf-8') as f2:
        f2.write("\n")


def read_file(def_path):
    file_path = input("请输入需纠错txt文件的路径：")
    if os.path.exists(file_path):
        print("文件", file_path, "存在，开始纠错。")
    else:
        print("文件", file_path, "不存在，选择默认文件", def_path, "开始纠错")
        file_path = def_path
    print("file_path = ", file_path)
    return file_path


def add_confusion_set(conf_table, cor_path, sent):
    while input("若仍有错判词汇，请输入 Y 添加到混淆集以防误杀：") == 'Y':
        conf_word = input("请输入被误判词汇：")
        with open(conf_table, 'a', encoding='utf-8') as f4:
            f4.write(conf_word + " " + conf_word + '\n')
        print("误判词 ", conf_word, "已加入自定义混淆集：", conf_table, " 中")
        print("重新纠错：")
        correct(sent, cor_path)
    print("纠错结果已保存在文件", cor_path, "中")


def keyword_TR(texts):
    # 引入TextRank关键词抽取接口1
    textrank = analyse.textrank
    # 抽取目标文章的关键字
    keywords = textrank(texts)
    print(type(keywords))
    # 输出抽取的关键字
    for keyword in keywords:
        print(keyword + "/")


def keyword_TF(texts):
    # 提取关键词：topK，关键词数量；withWeight：是否附带权重；allowPOS：是否限定词性
    keywords = jieba.analyse.extract_tags(texts, topK=20, withWeight=False, allowPOS=())
    print(keywords)
    # ===== 打开纠正结果记录 =====
    with open(correct_result, 'a+', encoding='utf-8') as f2:
        f2.write("1.文章的关键词为：\n")
        for i in range(1, 6):
            f2.write("(" + str(i) + ") " + keywords[i - 1] + " ")
        f2.write("\n")
        f2.write("\n")


def jb_cut(texts):
    # 中文分词
    seg = jieba.cut(texts)
    # 输出分词结果
    l1 = []
    for i in seg:
        l1.append(i)
    print(l1)
    # 中文词性标注
    seg2 = jieba.posseg.cut(texts)
    # 输出词性标注结果
    l2 = []
    for i in seg2:
        l2.append((i.word, i.flag))
    print(l2)


# 定义停词函数 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    stopwords = [line.strip() for line in open('./baidu_stopwords.txt', 'r', encoding='UTF-8').readlines()]
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word not in [" ", ",", "，", "?", "？", ".", "。", "!", "！", "、", ":", "：", "(", ")", "（", "）", ";", "；",
                            "\n", "\t"]:
                outstr += word
                outstr += " "
    return outstr


def create_LDA(file_path):
    with open(file_path, 'r', encoding='utf-8') as f2:
        texts = list(set(list(i.replace('\n', '').replace(' ', '') for i in list(f2.readlines()))))

    # 导入停用词
    stopwords = [line.strip() for line in open('./baidu_stopwords.txt', 'r', encoding='UTF-8').readlines()]

    # 分词
    result_fenci = []
    for i in texts:
        # print(i)
        if seg_depart(i) != '':
            result_fenci.append([i, seg_depart(i)])
    result_fenci = [i[1].split(' ')[:-1] for i in result_fenci]
    print(result_fenci)

    # 创建词典
    id2word = corpora.Dictionary(result_fenci)
    corpus = [id2word.doc2bow(sentence) for sentence in result_fenci]
    # print(corpus[:1])

    # 创建主题模型
    topics = []
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=10)
    for topic in lda_model.print_topics(num_words=5):
        # print(topic[1].split('"'))
        print(topic)
        topics.append(topic[1].split('"')[1])

    # 模型复杂度
    print('模型复杂度: ', lda_model.log_perplexity(corpus))

    # 主题一致性
    coherence_model_lda = CoherenceModel(model=lda_model, texts=result_fenci, dictionary=id2word, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    print('主题一致性得分: ', coherence_lda)

    # 最优主题数
    coherence_values = []
    model_list = []
    for num_topics in range(2, 41, 2):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
        model_list.append(lda_model)
        coherencemodel = CoherenceModel(model=lda_model, texts=result_fenci, dictionary=id2word, coherence='u_mass')
        coherence_values.append(round(coherencemodel.get_coherence(), 3))
    x = range(2, 41, 2)

    # 绘制曲线图
    # plt.plot(x, coherence_values)
    # plt.xlabel("Num Topics")
    # plt.ylabel("Coherence score")
    # plt.legend(("coherence_values"), loc='best')
    # plt.show()

    # 输出各主题数一致性结果
    best_model = 1
    max_cs = -999
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
        if max_cs < cv:
            max_cs = cv
            best_model = m
    print("coherence_values: ", best_model, max_cs)
    # 得到最优主题数
    optimal_model = model_list[int(best_model / 2) - 1]

    # 找到每句的最优主题
    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=result_fenci)

    # 基于重要性排序
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # 展示每句的最优主题
    df_dominant_topic.to_excel('./resultsdatas.xlsx', index=False)
    # print(df_dominant_topic.head(10))

    # 展示每个主题的代表性句子
    pd.options.display.max_colwidth = 100
    # 基于主题贡献度排序
    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
    # 加入代表性句子
    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                                axis=0)
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
    # print(sent_topics_sorteddf_mallet.head(10))

    # 判断输出主题数目
    topic_num = len(sent_topics_sorteddf_mallet['Keywords'])
    pre_num = 5
    if topic_num < pre_num:
        pre_num = topic_num

    # ===== 打开纠正结果记录 =====
    with open(correct_result, 'a+', encoding='utf-8') as f2:
        f2.write("3.文章的主题和对应的代表句为：\n")
        for i in range(0, pre_num):
            # 记录主题关键词
            print(sent_topics_sorteddf_mallet['Keywords'][i])
            pre_topic = sent_topics_sorteddf_mallet['Keywords'][i].split(',')
            f2.write("主题" + str(i) + "为: ")
            for j in range(0, 3):
                f2.write(pre_topic[j] + " ")
            f2.write("\n")
            # 记录代表句
            f2.write("该主题的代表性句子是: ")
            sens = sent_topics_sorteddf_mallet['Representative Text'][i]
            for sent in sens:
                f2.write(sent)
            f2.write("\n")
            f2.write("\n")
        f2.write("\n")


def format_topics_sentences(ldamodel, corpus, texts):
    # 初始化
    sent_topics_df = pd.DataFrame()
    # 获得最优主题
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # 提取主要主题
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # 最优主题
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # 加入原始文本
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


# ========================== 默认源文件路径 ========================
path = "./demo.txt"
# path = './poem.txt'

# 读入纠错文件
path = read_file(path)

# 纠错后文件路径；
correct_path = path[:-4] + "_correct.txt"

# 自定义混淆集路径
confusion_table = './my_custom_confusion.txt'

# 纠错结果路径
correct_result = path[:-4] + "_correct_result.txt"
print("correct_result: ", correct_result)
with open(correct_result, 'w+', encoding='utf-8') as f3:
    f3.write("")

# 读入源文件
f = open(path, "r", encoding='utf-8')
text = f.read()
f.close()

# ===== 1. jieba 提取关键词 =====
# keyword_TR(text)  # 基于 TextRank 方法
keyword_TF(text)  # 基于 TF-IDF 方法

# ===== 2. jieba 分词 =====
jb_cut(text)

# ===== 3. 中文错别字纠错 =====
# 断句
sentences = re.split(r"(['。！？!?.；;,，\n ])", text)
sentences.append("")
sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
# print(sentences)

# 中文纠错
correct(sentences, correct_path)

# 添加到混淆集
# add_confusion_set(confusion_table, correct_path, sentences)

# ===== 4. 创建主题模型 =====
create_LDA(path)
