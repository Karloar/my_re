import os
import platform
from gensim.models import Word2Vec
from myfuncs2 import get_trigger_list_from_sents
from myfuncs2 import get_resource_path
from myfuncs2 import load_data_en
from myfuncs2 import print_running_time
from myfuncs2 import Param
from myfuncs2 import get_trigger_neighbour_words_list
from myfuncs2 import get_trigger_neighbour_vector_list
from myfuncs2 import get_feature_vector_for_rnn
from myfuncs2 import get_label_by_entity_relation_list
from myfuncs2 import MyRNNClassifier


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
    train_data = get_feature_vector_for_rnn(
        trigger_neighbour_vector_list, vector_size
    )
    train_label = get_label_by_entity_relation_list(entity_relation_list, relation_list)

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
    test_data = get_feature_vector_for_rnn(
        trigger_neighbour_vector_list, vector_size
    )
    test_label = get_label_by_entity_relation_list(entity_relation_list, relation_list)
    print('classifying......')
    mrc = MyRNNClassifier(
        vector_size,
        len(relation_list),
        params.trigger_neighbour,
        vector_size * 2,
        batch_size=params.batch_size,
        keep_prob=0.7,
        use_bilstm=params.use_bilstm,
        use_attention=params.use_attention
    )
    mrc.fit(train_data, train_label)
    print('train accuracy:', mrc.accuracy(train_data, train_label))
    print('test accuracy:', mrc.accuracy(test_data, test_label))
    print(relation_list)
    print(mrc.report(test_data, test_label))
    mrc.close()
    
    
if __name__ == '__main__':
    main()
