import librosa
import os
import numpy as np
import pyaudio
import wave
import sys

def readCorpus(path):
    
    # 读入音乐节奏数据库
    all_data = np.load(path, allow_pickle=True)
    beat_database = all_data.item()
    
    return beat_database

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
            raw_db[audioName] = f

    # 保存音乐节奏数据库
    np.save(dbpath, raw_db)
    
# 语料库路径
corpus_path = './corpus'

# 数据表路径
dbPath = './beatDatabase_mfcc.npy'

inPara = sys.argv

if (len(inPara) < 2):
    print("默认语料库路径为：", corpus_path)
else:
    print("给定语料库路径为:", sys.argv[1])
    corpus_path = sys.argv[1]

updateCorpus(corpus_path, dbPath)
print("语料库已更新！")