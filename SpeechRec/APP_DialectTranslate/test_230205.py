#!/usr/bin/env python
# coding: utf-8

import utils.corpus as VCP
import utils.record as REC


def recComp():
    # 初始化录音器
    
    rec = REC.REC()
    path = rec.record_voice()
    
    return path


def Test(cPath, dbPath):
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

Test(corpus_path, dbPath)





