import utils.corpus as VCP
import utils.record as REC

def recComp():
    # 初始化录音器
    
    rec = REC.REC()
    path = rec.record_voice()
    
    return path

corpus_path = './corpus' # 语料库路径
dbPath = './beatDatabase.npy' # 数据表路径

testPath = ""
# 1 录音输入文件
testPath = recComp()

# 2 手动设置输入文件
#testPath = './input/20230203234857.wav'

# 比较最相似语音
vComp = VCP.voiceComparator(corpus_path, dbPath)
vComp.voiceCompare(testPath)