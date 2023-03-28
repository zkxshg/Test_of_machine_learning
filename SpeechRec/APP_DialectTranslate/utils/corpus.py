import librosa
import os
import numpy as np

from dtw import dtw
from numpy.linalg import norm
from numpy import array
import pyaudio
import wave

class voiceComparator:
    def __init__(self, corpus_path, dbPath):
        self.corpus_path = corpus_path
        self.dbPath = dbPath
    
    # 初始化语料库
    def initialCorpus(self):
        path = self.corpus_path
        
        # 音乐库位置
        audioList = os.listdir(path)

        raw_audioList = {}
        beat_database = {}

        for tmp in audioList:
            audioName = os.path.join(path, tmp)
            if audioName.endswith('.wav'):
                y, sr = librosa.load(audioName)
                # 跟踪歌曲的节奏点
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                # 对提取的节奏序列进行差分
                beat_frames = librosa.feature.delta(beat_frames, mode ='nearest')
                # 存入数据表
                beat_database[audioName] = beat_frames

        # 保存音乐节奏数据库
        np.save(self.dbPath, beat_database)

        return beat_database
    
    # 读入语料库
    def readCorpus(self):
        path = self.dbPath
        
        # 读入音乐节奏数据库
        all_data = np.load(path, allow_pickle=True)
        beat_database = all_data.item()

        return beat_database
    
    # 更新语料库
    def updateCorpus(self):
        path = self.corpus_path 
        dbpath = self.dbPath
        
        # 音乐库位置
        audioList = os.listdir(path)

        # 已保存序列的文件
        raw_db = readCorpus(dbPath)
        raw_files = raw_db.keys()

        for tmp in audioList:
            audioName = os.path.join(path, tmp)
            if audioName.endswith('.wav') and audioName not in raw_files:
                y, sr = librosa.load(audioName)
                # 跟踪歌曲的节奏点
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                # 对提取的节奏序列进行差分
                beat_frames = librosa.feature.delta(beat_frames,mode ='nearest')
                # 存入数据表
                beat_database[audioName] = beat_frames

        # 保存音乐节奏数据库
        np.save(dbpath, beat_database)
        
        return dbpath
    
    # 找出语料库中最接近的语音
    def voiceCompare(self, tPath):
        dbPath = self.dbPath
        
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
        
        return matched_song
    
    
    
      