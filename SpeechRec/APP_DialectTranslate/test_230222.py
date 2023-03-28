#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import os
import numpy as np
import sys
from dtw import dtw
from numpy.linalg import norm
from numpy import array
import pyaudio
import wave

import heapq


# In[2]:


def initialCorpus(path):
    # 音乐库位置
    audioList = os.listdir(path)

    raw_audioList = {}
    beat_database = {}

    for tmp in audioList:
        audioName = os.path.join(path, tmp)
        if audioName.endswith('.wav'):
            # 读入一维音频序列
            y, sr = librosa.load(audioName)
            # 提取 MFCC 特征
            f = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
            # 存入数据表
            beat_database[audioName] = f

    # 保存音乐节奏数据库
    np.save('beatDatabase_mfcc.npy', beat_database)
    
    return beat_database


# In[3]:


def readCorpus(path):
    
    # 读入音乐节奏数据库
    all_data = np.load(path, allow_pickle=True)
    beat_database = all_data.item()
    
    return beat_database


# In[4]:


def updateCorpus(path, dbpath):
    
    # 音乐库位置
    audioList = os.listdir(path)
    
    # 已保存序列的文件
    raw_db = readCorpus(dbPath)
    raw_files = raw_db.keys()
    
    for tmp in audioList:
        audioName = os.path.join(path, tmp)
        if audioName.endswith('.wav') and audioName not in raw_files:
            y, sr = librosa.load(audioName)
            # 提取 MFCC 特征
            f = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
            # 存入数据表
            beat_database[audioName] = f

    # 保存音乐节奏数据库
    np.save(dbpath, beat_database)


# In[12]:


def normlize(data):
    n_mean = np.mean(data, axis=0)
    n_std  = np.std(data, axis=0)
    
    norm_data = np.divide(np.subtract(data, n_mean), n_std)
    return norm_data


# In[5]:


def voiceCompare_quick(dbPath, tPath):
    
    # 读入语料库
    all_data = np.load(dbPath, allow_pickle=True)
    beat_database = all_data.item()

    # 读入要识别的录音
    y, sr = librosa.load(tPath)

    # 识别录音的节奏序列
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_frames = librosa.feature.delta(beat_frames,mode ='nearest')
    x = array(beat_frames).reshape(-1, 1)

    # 将待识别的录音序列与语料库中语音逐一做DTW对比
    compare_result = {}
    
    for songID in beat_database.keys():
        y = beat_database[songID]
        y = array(y).reshape(-1, 1)
        
        dist = dtw(x, y).distance
        # print('两段话的差异程度为： ', songID.split("\\")[1], ": ", dist)
        
        compare_result[songID] = dist

    matched_song = min(compare_result, key=compare_result.get)
    print("最接近的录音是：", matched_song)


# In[18]:


from sklearn import preprocessing

def voiceCompare(dbPath, tPath):
    # 最大检索数
    aimNum = 20
    
    # 读入语料库
    all_data = np.load(dbPath, allow_pickle=True)
    beat_database = all_data.item()

    # ==== 读入要识别的录音 ====
    y, sr = librosa.load(tPath)

    # 提取录音的 MFCC 特征
    x = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10).T  # n1 * 10
    lenx = len(x)
    
    # 标准化
    for i in range(0, lenx):
        # x[i] = preprocessing.minmax_scale(x[i])
        x[i] = normlize(x[i])

    # ==== 将待识别的录音序列与语料库中语音逐一做DTW对比 ====
    
    # heap for [dist, 时间段，文件名]
    heap = []
    heapq.heapify(heap)  
    
    for songID in beat_database.keys():
        # 取出文件名对应的 mfcc 序列
        y = beat_database[songID].T
        
        leny = len(y) # n2 * 10 
        print(leny)
        
        # 标准化
        for i in range(0, leny):
            # y[i] = preprocessing.minmax_scale(y[i])
            y[i] = normlize(y[i])
        
        for tp in range(0, leny - lenx):
            # *加速* 设定距离上限
            full = False  # 堆是否已满
            dist_UB = -10000  # DTW 距离上限
            overBound = False  # 是否过限
            
            if (len(heap) >= aimNum):
                full = True
                dist_UB = -heap[0][0]  # heap top (biggest) DTW dist as UB  
                
            # 计算 DTW(y[tp : tp + lenx])
            total_dist = 0
            # euc_dist = 0
            
            for i in range(0, lenx):
                # DTW dist
                total_dist += dtw(x[i], y[tp + i], distance_only=False).distance
                # Euclidean dist
                # euc_dist += np.linalg.norm(p1 - p2)
                
                # *加速* 超过上限直接取消
                if (full and total_dist > dist_UB):
                    overBound = True
                    break
            
            # *加速* 超过上限
            if (overBound):
                continue
            
            # 入栈
            tupleY = (-total_dist, tp, songID) # dtw 距离加负数转为大根堆
            
            heapq.heappush(heap, tupleY)
            if (len(heap) > aimNum):
                heapq.heappop(heap)
            
            print(tupleY)
            
        # end for
        
        # 处理同名短间隔问题
        
        
    return heap


# In[25]:


def getTimePoint_dense(dbPath, tPath, vheap):
    res_num = 20 # 定义取出前 res_num 位的结果作为识别结果
    
    # 读入语料库
    all_data = np.load(dbPath, allow_pickle=True)
    beat_database = all_data.item()
    
    # 得到要识别的录音时长
    tTime = librosa.get_duration(filename=tPath)
    
    # 提取前 res_num 个相似的片段并输出对应时间段
    similar_n = heapq.nlargest(res_num, vheap)
    
    print("开始输出相似片段：")
    
    for i in range(0, res_num):
        music_name = similar_n[i][2]  # 录音文件名
        music_time = librosa.get_duration(filename=music_name)  # 录音时长
        
        music_pos = similar_n[i][1]  # 时间段所在帧数
        music_all = len(beat_database[music_name][0])  # 录音总帧数

        frag_st = music_time / music_all * music_pos  # 时间段起点
        frag_en = frag_st + tTime  # 时间段终点
        
        # print(music_name, music_time, music_pos, music_all, frag_st)
        # print("相似度第", i + 1, "位的为文件 ", music_name, "的 ", '%.2f' % frag_st, "到", '%.2f' % frag_en, "秒")
        
        print(music_name, ",", '%.2f' % frag_st, "秒,", '%.2f' % frag_en, "秒")


# In[26]:


def getTimePoint(dbPath, tPath, vheap):
    # 读入语料库
    all_data = np.load(dbPath, allow_pickle=True)
    beat_database = all_data.item()
    
    # 得到要识别的录音时长
    tTime = librosa.get_duration(filename=tPath)
    
    heapq.nlargest(20, vheap)
    
    # ====== 对 vheap 进行去重 ======
    # 取出文件名
    name_set = set()
    for tp in vheap:
        name_set.add(tp[2])
    # print(name_set)
    
    # 合并下标差小于5的片段
    sheap = []
    for name in name_set:
        # 按下标排序
        nList = [x for x in vheap if x[2] == name]
        sortL = sorted(nList, key=lambda t:t[1])
        
        # 去重
        for tp in sortL:
            if len(sheap) < 1 or sheap[-1][2] != name or abs(sheap[-1][1] - tp[1]) > 5:
                sheap.append(tp)
            else:  
                if (sheap[-1][0] < tp[0]): 
                    sheap[-1] = tp  # 保留距离较小项

    # print(sheap)
    # 提取相似片段并输出对应时间段
    similar_n = sheap
    
    print("开始输出相似片段：")
    
    for i in range(0, len(sheap)):
        music_name = similar_n[i][2]  # 录音文件名
        music_time = librosa.get_duration(filename=music_name)  # 录音时长
        
        music_pos = similar_n[i][1]  # 时间段所在帧数
        music_all = len(beat_database[music_name][0])  # 录音总帧数

        frag_st = music_time / music_all * music_pos  # 时间段起点
        frag_en = frag_st + tTime  # 时间段终点
        
        # print(music_name, music_time, music_pos, music_all, frag_st)
        # print("相似度第", i + 1, "位的为文件 ", music_name, "的 ", '%.2f' % frag_st, "到", '%.2f' % frag_en, "秒")
        
        print(music_name, ",", '%.2f' % frag_st, "秒,", '%.2f' % frag_en, "秒")


# In[20]:


# 语料库路径
corpus_path = './corpus'

# 数据表路径
dbPath = './beatDatabase_mfcc.npy';

# test file path
#testPath = './input/00415250-前5s.wav'
testPath = './input/00430105-hou5s.wav'


# In[28]:


# 1 初始化语料序列库
# beatDB = initialCorpus(corpus_path)

# 2 更新语料库中新音乐文件的序列
# updateCorpus(corpus_path, dbPath)

# 3 读入语料序列库
# beat_database = readCorpus(dbPath)

# vheap = voiceCompare(dbPath, testPath)


# In[29]:


# getTimePoint(dbPath, testPath, vheap)


# In[ ]:


inPara = sys.argv

if (len(inPara) < 2):
    print("请输入待识别录音文件路径！")
else:
    if (len(inPara) > 2):
        print("给定语料库路径为:", sys.argv[2])
        corpus_path = sys.argv[2]
    else:
        print("默认语料库路径为：", corpus_path)
    
    if (len(inPara) > 3):
        print("给定数据表路径为:", sys.argv[3])
        dbPath = sys.argv[3]
    else:
        print("默认数据表路径为：", dbPath)
    
    testPath = sys.argv[1]
    vheap = voiceCompare(dbPath, testPath)
    getTimePoint(dbPath, testPath, vheap)
  


# In[ ]:




