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
from gensim.models import Word2Vec
from sklearn.neural_network import MLPClassifier



train_file = get_resource_path("data/TRAIN_FILE.TXT")
test_file = get_resource_path("data/TEST_FILE_FULL.TXT")
word2vec_model_file = 'en_word2vec.model'
train_data_pkl = 'train_data.pkl'
train_label_pkl = 'train_label.pkl'
test_data_pkl = 'test_data.pkl'
test_label_pkl = 'test_label.pkl'


def get_trigger_vector(trigger_list, model, length):
    trigger_vector = np.zeros((1, length))
    for word in trigger_list[0]:
        if word in model:
            trigger_vector += model[word].reshape(1, length)
    trigger_vector /= len(trigger_list[0])
    for i in range(1, len(trigger_list)):
        tv = np.zeros((1, length))
        for word in trigger_list[i]:
            if word in model:
                tv += model[word].reshape(1, length)
        tv /= len(trigger_list[i])
        trigger_vector = np.concatenate([trigger_vector, tv], axis=0)
    return trigger_vector
    

@print_running_time
def main():
    train_sent_list, train_entity_relation_list = load_data_en(train_file)
    test_sent_list, test_entity_relation_list = load_data_en(test_file)
    if os.path.exists(word2vec_model_file):
        model = Word2Vec.load(word2vec_model_file)
    else:
        model = Word2Vec(sentences=train_sent_list, size=200)
        model.save(word2vec_model_file)
    relation_type_file = "relation_type.pkl"
    if not os.path.exists(relation_type_file):
        relation_type_list = set()
        for train_entity_relation in train_entity_relation_list:
            relation_type_list.add(train_entity_relation[1])
        pkl.dump(list(relation_type_list), open(relation_type_file, "wb"))
    else:
        relation_type_list = pkl.load(open(relation_type_file, "rb"))
    
    if os.path.exists(train_data_pkl) and os.path.exists(train_label_pkl) and os.path.exists(test_data_pkl) and os.path.exists(test_label_pkl):
        train_data = pkl.load(open(train_data_pkl, 'rb'))
        train_label = pkl.load(open(train_label_pkl, 'rb'))
        test_data = pkl.load(open(test_data_pkl, 'rb'))
        test_label = pkl.load(open(test_label_pkl, 'rb'))
    else:
        with StanfordCoreNLP('http://localhost', port=9000, lang='en') as nlp:
    
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
                if trigger[2] == 0 and len(word_list) >= 3:
                    trigger_list_element.extend([word_list[trigger[2]], word_list[trigger[2]+1], word_list[trigger[2]+2]])
                elif trigger[2] == len(word_list) - 1 and len(word_list) >= 3:
                    trigger_list_element.extend([word_list[trigger[2]-2], word_list[trigger[2]-1], word_list[trigger[2]]])
                elif len(word_list) >= 3:
                    trigger_list_element.extend([word_list[trigger[2]-1], word_list[trigger[2]], word_list[trigger[2]+1]])
                trigger_list.append(trigger_list_element)
            train_label = np.array(train_label_vector).reshape(len(train_sent_list), 1)
            train_data = get_trigger_vector(trigger_list, model, 200)
            pkl.dump(train_label, open(train_label_pkl, 'wb'))
            pkl.dump(train_data, open(train_data_pkl, 'wb'))
            print('train data is over')
            # 测试集
            trigger_list = []
            test_label_vector = []
            for sent, entity_relation in zip(test_sent_list, test_entity_relation_list):

                test_label_vector.append(relation_type_list.index(entity_relation[1]))

                word_list = nlp.word_tokenize(sent)
                dependency_tree = nlp.dependency_parse(sent)
                postags = [x[1] for x in nlp.pos_tag(sent)]
                q1_set = get_qfset(entity_relation[0], word_list)
                q2_set = get_qfset(entity_relation[2], word_list)
                i_vector = get_I_vector_by_qfset(q1_set, q2_set, word_list, dependency_tree)
                trigger_candidate = get_trigger_candidate(word_list, i_vector, postags, q1_set, q2_set, style="stanfordcorenlp")
                trigger = get_trigger_by_ap_cluster(trigger_candidate, i_vector)
                trigger_list_element = []
                if trigger[2] == 0 and len(word_list) >= 3:
                    trigger_list_element.extend([word_list[trigger[2]], word_list[trigger[2]+1], word_list[trigger[2]+2]])
                elif trigger[2] == len(word_list) - 1 and len(word_list) >= 3:
                    trigger_list_element.extend([word_list[trigger[2]-2], word_list[trigger[2]-1], word_list[trigger[2]]])
                elif len(word_list) >= 3:
                    trigger_list_element.extend([word_list[trigger[2]-1], word_list[trigger[2]], word_list[trigger[2]+1]])
                trigger_list.append(trigger_list_element)
            test_label = np.array(test_label_vector).reshape(len(test_sent_list), 1)
            test_data = get_trigger_vector(trigger_list, model, 200)
            pkl.dump(test_label, open(test_label_pkl, 'wb'))
            pkl.dump(test_data, open(test_data_pkl, 'wb'))
            print('test data is over')
    
    mlp = MLPClassifier(max_iter=5000, learning_rate_init=0.00001, hidden_layer_sizes=(100, 100, 100), activation='tanh')
    mlp.fit(train_data, train_label.ravel())
    print(mlp.score(train_data, train_label.ravel()))
    print(mlp.score(test_data, test_label.ravel()))


if __name__ == '__main__':
    main()