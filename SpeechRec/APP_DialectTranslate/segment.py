#!/usr/bin/env python
# coding: utf-8

# In[4]:


from pydub import AudioSegment
import datetime
import sys


# In[5]:


def musicSegment(path, st, en):
    # 读入文件
    song = AudioSegment.from_wav(path)
    st = int(st)
    en = int(en)
    if (st >= len(song) or en >= len(song)):
        print("给定时间超出音乐长度，截取失败！")
        return
    
    # 用当前时间作为保存文件名
    tmpTime = datetime.datetime.now() 
    save_time = tmpTime.strftime("%Y%m%d%H%M%S")
    save_path = save_time + ".wav"
    
    # 保存文件
    song[st : en].export(save_path, format="wav")
    print("文件已保存在", save_path)


# In[6]:


# path = "./input/00415250.wav"
inPara = sys.argv

if (len(inPara) < 2):
    print("请输入待识别录音文件路径！")
else:
    if (len(inPara) > 2):
        print("给定起始时间为:", sys.argv[2], "ms")
        st = sys.argv[2]
    else:
        print("请输入片段的起始时间！")
    
    if (len(inPara) > 3):
        print("给定结束时间为:", sys.argv[3])
        en = sys.argv[3]
    else:
        print("请输入片段的结束时间！")
    
    testPath = sys.argv[1]
    
    if (len(inPara) > 3):
        musicSegment(testPath, st, en)


# In[ ]:




