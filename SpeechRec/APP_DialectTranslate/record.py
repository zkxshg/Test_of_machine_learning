import speech_recognition as sr
import datetime

class REC:
    def __init__(self):  # 模型初始化
        self.sample_rate = 48000
        self.data_size = 8192
        
    def record_voice(self):
        # 设置采样率
        sample_rate = 48000
        data_size = 8192

        # 初始化
        recog = sr.Recognizer()

        with sr.Microphone(sample_rate = sample_rate, chunk_size = data_size) as source:
            # recog.adjust_for_ambient_noise(source)
            print('请开始发言: ')
            speech = recog.listen(source)

        savepath = "NULL"

        try:
            # text = recog.recognize_google(speech, language='zh-CN')
            # print('你说的是: ' + text)

            # 用当前时间作为保存文件名
            tmpTime = datetime.datetime.now() 
            save_time = tmpTime.strftime("%Y%m%d%H%M%S")
            save_path = "./input/" + save_time + ".wav"

            # write audio to a WAV file
            with open(save_path, "wb") as f:
                f.write(speech.get_wav_data()) 
                print("录音文件已保存在:", save_path)

                savepath = save_path

        except sr.UnknownValueError:
            print('听不到你的声音！')
        except sr.RequestError as e:
            print("程序发生错误; {}".format(e))

        return savepath