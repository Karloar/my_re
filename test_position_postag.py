import os
import platform
from myfuncs2 import get_trigger_list_from_sents
from myfuncs2 import get_resource_path
from myfuncs2 import load_data_en
from myfuncs2 import print_running_time
from myfuncs2 import Param
from myfuncs2 import get_trigger_neighbour_words_list
from myfuncs2 import get_trigger_neighbour_vector_list
from myfuncs2 import get_feature_vector_for_classify
from myfuncs2 import get_label_by_entity_relation_list
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


model_dir = '/Users/karloar/Documents/other/ltp_data_v3.4.0'
if platform.system() == 'Windows':
    model_dir = r'E:\ltp_data'
train_file = get_resource_path("data/TRAIN_FILE.TXT")
test_file = get_resource_path("data/TEST_FILE_FULL.TXT")
train_trigger_list_pkl = 'saved/SemEval2010/train_trigger_list.pkl'
test_trigger_list_pkl = 'saved/SemEval2010/test_trigger_list.pkl'


@print_running_time
def main():
    print('processing train data......')
    sents, entity_relation_list = load_data_en(train_file, remove_other=False)
    relation_list = sorted(list(set(x[1] for x in entity_relation_list)))
    trigger_list, word_lists = get_trigger_list_from_sents(
        sents,
        entity_relation_list,
        os.path.join(os.getcwd(), train_trigger_list_pkl)
    )
    train_data = get_feature_vector_for_classify(word_lists, trigger_list, entity_relation_list)
    train_label = get_label_by_entity_relation_list(entity_relation_list, relation_list)

    print('processing test data......')
    sents, entity_relation_list = load_data_en(test_file, remove_other=False)
    trigger_list, word_lists = get_trigger_list_from_sents(
        sents,
        entity_relation_list,
        os.path.join(os.getcwd(), test_trigger_list_pkl)
    )
    test_data = get_feature_vector_for_classify(word_lists, trigger_list, entity_relation_list)
    test_label = get_label_by_entity_relation_list(entity_relation_list, relation_list)
    # color = [cm.get_cmap('jet')(x) for x in np.linspace(0, 1, 10)]
    # train_data_ = np.argmax(train_label, axis=1)
    # for i in range(len(train_data)):
    #     plt.scatter(train_data[i, 0], train_data[i, 1], color=color[train_data_[i]])
    # plt.show()
    mlp = MLPClassifier(hidden_layer_sizes=(100000,))
    mlp.fit(train_data, train_label)
    test_pred = mlp.predict(test_data)
    print('train accuracy:', mlp.score(train_data, train_label))
    print('test accuracy:', mlp.score(test_data, test_label))
    print(metrics.classification_report(test_label, test_pred))


if __name__ == '__main__':
    main()