import os
import platform
from gensim.models import Word2Vec
import numpy as np
from myfuncs2 import get_trigger_list_from_sents
from myfuncs2 import get_resource_path
from myfuncs2 import load_data_en
from myfuncs2 import print_running_time
from myfuncs2 import Param
from myfuncs2 import get_trigger_neighbour_words_list
from myfuncs2 import get_trigger_neighbour_vector_list
from myfuncs2 import get_feature_vector_for_nn2
from myfuncs2 import get_label_by_entity_relation_list
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


model_dir = '/Users/karloar/Documents/other/ltp_data_v3.4.0'
if platform.system() == 'Windows':
    model_dir = r'E:\ltp_data'
train_file = get_resource_path("data/TRAIN_FILE.TXT")
test_file = get_resource_path("data/TEST_FILE_FULL.TXT")
word2vec_model_file = os.path.join(model_dir, 'wiki_en_vector_vec.model')

vector_size = 200
train_trigger_list_pkl = 'saved/SemEval2010/train_trigger_list.pkl'
test_trigger_list_pkl = 'saved/SemEval2010/test_trigger_list.pkl'


@print_running_time
def main():

    # 设置参数
    params = Param()
    params.remove_other = True

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
    train_data = get_feature_vector_for_nn2(word_lists, trigger_list, vector_size, model, entity_relation_list)
    train_label = get_label_by_entity_relation_list(entity_relation_list, relation_list)

    # 处理测试数据
    print('processing test data......')
    sents, entity_relation_list = load_data_en(test_file, remove_other=params.remove_other)
    trigger_list, word_lists = get_trigger_list_from_sents(
        sents,
        entity_relation_list,
        os.path.join(os.getcwd(), test_trigger_list_pkl)
    )
    test_data = get_feature_vector_for_nn2(word_lists, trigger_list, vector_size, model, entity_relation_list)
    test_label = get_label_by_entity_relation_list(entity_relation_list, relation_list)
 
    print('classifying......')
    mlp = MLPClassifier(max_iter=100000, learning_rate_init=0.001, hidden_layer_sizes=((vector_size+4) * 2,) * 10)
    mlp.fit(train_data, train_label)
    test_pred = mlp.predict(test_data)
    print('train accuracy:', mlp.score(train_data, train_label))
    print('test accuracy:', mlp.score(test_data, test_label))
    print(metrics.classification_report(test_label, test_pred))


if __name__ == '__main__':
    main()
