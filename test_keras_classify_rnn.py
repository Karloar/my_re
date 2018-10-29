from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.wrappers.scikit_learn import KerasClassifier
from gensim.models import Word2Vec
import numpy as np
import platform
import os
from sklearn import metrics
from myfuncs2 import get_resource_path
from myfuncs2 import Param
from myfuncs2 import print_running_time
from myfuncs2 import load_data_en
from myfuncs2 import get_trigger_list_from_sents
from myfuncs2 import get_trigger_neighbour_words_list
from myfuncs2 import get_trigger_neighbour_vector_list
from myfuncs2 import get_feature_vector_for_rnn3
from myfuncs2 import get_label_by_entity_relation_list


model_dir = '/Users/karloar/Documents/other/ltp_data_v3.4.0'
if platform.system() == 'Windows':
    model_dir = r'E:\ltp_data'
train_file = get_resource_path("data/TRAIN_FILE.TXT")
test_file = get_resource_path("data/TEST_FILE_FULL.TXT")
word2vec_model_file = os.path.join(model_dir, 'wiki_en_vector_vec.model')

vector_size = 200 
train_trigger_list_pkl = 'saved/SemEval2010/train_trigger_list.pkl'
test_trigger_list_pkl = 'saved/SemEval2010/test_trigger_list.pkl'


def create_model():
    n_hidden_input = 400
    n_classes = 11
    model = Sequential()
    model.add(Dense(n_hidden_input))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(n_hidden_input * 2, return_sequences=False)))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


@print_running_time
def main():
    # 设置参数
    params = Param()
    params.trigger_neighbour = 9
    params.batch_size = 200
    params.remove_punc = False
    params.remove_other = True
    params.use_bilstm = False
    params.use_attention = False
    params.trigger_neighbour_type = 2

    # 加载word2vec模型
    model = Word2Vec.load(word2vec_model_file)

    # 处理训练数据
    print('processing train data......')
    sents, entity_relation_list = load_data_en(train_file, remove_other=params.remove_other)
    relation_list = sorted(list(set(x[1] for x in entity_relation_list)))
    trigger_list, word_lists = get_trigger_list_from_sents(
        sents,
        entity_relation_list,
        os.path.join(os.getcwd(), train_trigger_list_pkl)
    )
    trigger_neighbour_words_list = get_trigger_neighbour_words_list(
        trigger_list,
        word_lists,
        params.trigger_neighbour,
        remove_punc=params.remove_punc,
        type=params.trigger_neighbour_type
    )
    trigger_neighbour_vector_list = get_trigger_neighbour_vector_list(
        trigger_neighbour_words_list, model, vector_size
    )
    train_data = get_feature_vector_for_rnn3(
        trigger_neighbour_vector_list, vector_size
    )
    train_label = np.argmax(get_label_by_entity_relation_list(entity_relation_list, relation_list), axis=1)

    # 处理测试数据
    print('processing test data......')
    sents, entity_relation_list = load_data_en(test_file, remove_other=params.remove_other)
    trigger_list, word_lists = get_trigger_list_from_sents(
        sents,
        entity_relation_list,
        os.path.join(os.getcwd(), test_trigger_list_pkl)
    )
    trigger_neighbour_words_list = get_trigger_neighbour_words_list(
        trigger_list,
        word_lists,
        params.trigger_neighbour,
        remove_punc=params.remove_punc,
        type=params.trigger_neighbour_type
    )
    trigger_neighbour_vector_list = get_trigger_neighbour_vector_list(
        trigger_neighbour_words_list, model, vector_size
    )
    test_data = get_feature_vector_for_rnn3(
        trigger_neighbour_vector_list, vector_size
    )
    test_label = np.argmax(get_label_by_entity_relation_list(entity_relation_list, relation_list), axis=1)
    print('classifying......')
    estimator = KerasClassifier(build_fn=create_model, epochs=100, batch_size=200, verbose=True)
    estimator.fit(train_data, train_label)
    test_pred = estimator.predict(test_data)
    print('train score:', estimator.score(train_data, train_label))
    print('test score:', estimator.score(test_data, test_label))
    print(metrics.classification_report(test_label, test_pred))


if __name__ == '__main__':
    main()
