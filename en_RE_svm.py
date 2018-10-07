from myfuncs import get_resource_path
from myfuncs import print_running_time
from myfuncs import load_data_en
from myfuncs import get_qfset
from myfuncs import get_I_vector_by_qfset
from myfuncs import get_trigger_candidate
from myfuncs import get_trigger_by_ap_cluster
from stanfordcorenlp import StanfordCoreNLP
import pickle as pkl
import os
import numpy as np
'''
向量维度过大，代码无法运行完
'''


train_file = get_resource_path("data/TRAIN_FILE.TXT")
test_file = get_resource_path("data/TEST_FILE_FULL.TXT")


def get_trigger_vector(trigger_list, vocab_list):
    trigger = trigger_list[0]
    n = len(vocab_list)
    trigger_vector = np.zeros((1, n))
    for word in trigger:
        if word in vocab_list:
            trigger_vector[0, vocab_list.index(word)] = 1
    for i in range(1, len(trigger_list)):
        t_v = np.zeros((1, n))
        for word in trigger_list[i]:
            if word in vocab_list:
                t_v[0, vocab_list.index(word)] = 1
        trigger_vector = np.concatenate([trigger_vector, t_v], axis=0)
    return trigger_vector
    


@print_running_time
def main():
    train_sent_list, train_entity_relation_list = load_data_en(train_file)
    test_sent_list, test_entity_relation_list = load_data_en(test_file)
    with StanfordCoreNLP('http://localhost', port=9000, lang='en') as nlp:

        # 加载词典
        vocab_list_file = "vocab_list.pkl"
        if not os.path.exists(vocab_list_file):
            vocab_list = list()
            for sent in train_sent_list:
                vocab_list.extend(nlp.word_tokenize(sent))
                vocab_list = list(set(vocab_list))
            pkl.dump(vocab_list, open(vocab_list_file, "wb"))
        else:
            vocab_list = pkl.load(open(vocab_list_file, "rb"))
        
        relation_type_file = "relation_type.pkl"
        if not os.path.exists(relation_type_file):
            relation_type_list = set()
            for train_entity_relation in train_entity_relation_list:
                relation_type_list.add(train_entity_relation[1])
            pkl.dump(list(relation_type_list), open(relation_type_file, "wb"))
        else:
            relation_type_list = pkl.load(open(relation_type_file, "rb"))


        trigger_list = []
        train_label_vector = []
        # 根据PageRank得到trigger, 同时构建训练数据
        for sent, entity_relation in zip(train_sent_list, train_entity_relation_list):

            train_label_vector.append(relation_type_list.index(entity_relation[1]))

            word_list = nlp.word_tokenize(sent)
            dependency_tree = nlp.dependency_parse(sent)
            postags = [x[1] for x in nlp.pos_tag(sent)]
            q1_set = get_qfset(entity_relation[0], word_list)
            q2_set = get_qfset(entity_relation[2], word_list)
            i_vector = get_I_vector_by_qfset(q1_set, q2_set, word_list, dependency_tree)
            trigger_candidate = get_trigger_candidate(word_list, i_vector, postags, q1_set, q2_set, style="stanfordcorenlp")
            trigger = get_trigger_by_ap_cluster(trigger_candidate, i_vector)
            trigger_list_element = []
            if trigger[2] - 1 >= 0:
                trigger_list_element.append(word_list[trigger[2]-1])
            trigger_list_element.append(trigger[0])
            if trigger[2] + 1 < len(word_list):
                trigger_list_element.append(word_list[trigger[2]+1])
            trigger_list.append(trigger_list_element)
        train_label_vector = np.array(train_label_vector).reshape(len(train_sent_list), 1)
        trigger_vector = get_trigger_vector(trigger_list, vocab_list)
        print(train_label_vector.shape)
        print(trigger_vector.shape)



if __name__ == '__main__':
    main()