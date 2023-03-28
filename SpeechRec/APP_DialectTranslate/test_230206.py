#!/usr/bin/env python
# coding: utf-8

import utils.corpus as VCP
import utils.record as REC
import sys

def recComp():
    # 初始化录音器
    
    rec = REC.REC()
    path = rec.record_voice()
    
    return path


def Test(testPath, corpus_path, dbPath):
    # 比较最相似语音
    vComp = VCP.voiceComparator(corpus_path, dbPath)
    vComp.voiceCompare(testPath)

def Test_Rec(cPath, dbPath):
    corpus_path = cPath
    
    ifListen = input("指定录音文件路径请输入 1, 其他将开始录音：")
    
    if (ifListen == '1'):
        testPath = input("请输入待识别图像文件夹路径：")
    else:
        testPath = recComp()
        
    # 比较最相似语音
    vComp = VCP.voiceComparator(corpus_path, dbPath)
    vComp.voiceCompare(testPath)
    
corpus_path = './corpus' # 语料库路径
dbPath = './beatDatabase.npy' # 数据表路径

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
        corpus_path = sys.argv[3]
    else:
        print("默认数据表路径为：", dbPath)
        
    Test(sys.argv[1], corpus_path, dbPath)





