import os
import platform
from gensim.models import Word2Vec
import numpy as np
from myfuncs import get_trigger_neighbour_list_from_sents
from myfuncs import get_resource_path
from myfuncs import load_data_en
from myfuncs import print_running_time
from myfuncs import get_trigger_neighbour_vector_list
from myfuncs import get_feature_vector_for_rnn
from myfuncs import Param
from myfuncs import get_label_by_entity_relation_list
from myfuncs import MyRNNClassifier


model_dir = '/Users/karloar/Documents/other/ltp_data_v3.4.0'
if platform.system() == 'Windows':
    model_dir = r'E:\ltp_data'
train_file = get_resource_path("data/TRAIN_FILE.TXT")
test_file = get_resource_path("data/TEST_FILE_FULL.TXT")
word2vec_model_file = os.path.join(model_dir, 'wiki_en_vector_vec.model')

vector_size = 200
train_neighbour_words_pkl = 'saved/train_neighbour_words.pkl'
test_neighbour_words_pkl = 'saved/test_neighbour_words.pkl'


@print_running_time
def main():

    # 加载word2vec模型
    model = Word2Vec.load(word2vec_model_file)

    # 设置参数
    params = Param()
    params.trigger_neighbour = 9

    # 处理训练数据
    print('processing train data......')
    sents, entity_relation_list = load_data_en(train_file)
    relation_list = sorted(list(set(x[1] for x in entity_relation_list)))
    trigger_neighbour_list = get_trigger_neighbour_list_from_sents(
        sents, entity_relation_list,
        os.path.join(os.getcwd(), train_neighbour_words_pkl),
        params=params
    )

    trigger_neighbour_vector_list = get_trigger_neighbour_vector_list(
        trigger_neighbour_list, model, vector_size
    )
    train_data = get_feature_vector_for_rnn(
        trigger_neighbour_vector_list, vector_size
    )
    train_label = get_label_by_entity_relation_list(entity_relation_list, relation_list)

    # 处理训练数据
    print('processing test data......')
    sents, entity_relation_list = load_data_en(test_file)
    trigger_neighbour_list = get_trigger_neighbour_list_from_sents(
        sents, entity_relation_list,
        os.path.join(os.getcwd(), test_neighbour_words_pkl),     
        params=params
    )
    trigger_neighbour_vector_list = get_trigger_neighbour_vector_list(
        trigger_neighbour_list, model, vector_size
    )
    test_data = get_feature_vector_for_rnn(
        trigger_neighbour_vector_list, vector_size
    )
    test_label = get_label_by_entity_relation_list(entity_relation_list, relation_list)
    print('classifying......')
    mrc = MyRNNClassifier(vector_size, len(relation_list), params.trigger_neighbour, vector_size*2, keep_prob=0.7, use_attention=False)
    mrc.fit(train_data, train_label)
    print('train accuracy:', mrc.accuracy(train_data, train_label))
    print('test accuracy:', mrc.accuracy(test_data, test_label))
    print(relation_list)
    print(mrc.report(test_data, test_label))
    mrc.close()
    

if __name__ == '__main__':
    main()
