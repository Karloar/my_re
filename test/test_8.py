import pickle
from gensim.models import Word2Vec

model_path = r'e:\ltp_data\wiki_model'

model = Word2Vec.load(model_path)

word_list = ['上传', '好友', '微博', '笑', '搞笑', '翻', '网友', '鲜有人']
for word in word_list:
    if word in model:
        print(word, len(model[word]))