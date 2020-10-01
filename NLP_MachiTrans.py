# pip install translate
from translate import Translator
print(Translator(to_lang="english").translate("你好"))

# 英文翻译为中文
# translator= Translator(from_lang="english",to_lang="chinese")
translator= Translator(from_lang="english",to_lang="zh")
translation = translator.translate("Translate is a simple but powerful translation tool \
written in python with support for multiple translation providers.")
print(translation)

# 中文翻译为英文
translator= Translator(from_lang="zh",to_lang="english")
translation = translator.translate("机器翻译将中文翻译为英文")
print(translation)

# pip install googletrans
from googletrans import Translator
googletranslator = Translator(service_urls=['translate.google.cn'])
googletranslator.translate('Hello World', dest='zh-CN').text

# 英译汉
result = googletranslator.translate('Googletrans is a free and unlimited python library\
that implemented Google Translate API. ', dest='zh-CN').text
print(result)

# 汉译英
result = googletranslator.translate('今天天气不错', dest='en').text
print(result)

# 英译日
result = googletranslator.translate('Hello', dest='ja').text
print(result)

# 英译韩
result = googletranslator.translate('Hello', dest='ko').text
print(result)

# 语言检测
print(googletranslator.detect('この文章は日本語で書かれました。').lang)
print(googletranslator.detect('自然语言处理课程').lang)
print(googletranslator.detect('How are you').lang)
# 置信度
googletranslator.detect('この文章は日本語で書かれました。').confidence

# 模型参数
transModel = googletranslator.translate('你好')
print(transModel.src)
print(transModel.dest)
print(transModel.text)

# 批量处理
translations = googletranslator.translate(['The quick brown fox', 'jumps over', 'the lazy dog'], dest='ko') 
for translation in translations:
    print(translation.origin, ' -> ', translation.text)
