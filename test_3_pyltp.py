from myfuncs import page_rank, get_I_vector, get_relation_word_top_n
from myfuncs import arcs_to_dependency_tree
from myfuncs import get_person_entity_set, get_modifier_set
from pyltp import Segmentor, NamedEntityRecognizer, Parser, Postagger
import os
import platform


cwd = os.getcwd()
model_dir = '/Users/karloar/Documents/other/ltp_data_v3.4.0'
if platform.system() == 'Windows':
    model_dir = r'E:\ltp_data'

cws_model = os.path.join(model_dir, 'cws.model')
cwd_dict = os.path.join(cwd, 'dict.txt')
pos_model = os.path.join(model_dir, 'pos.model')
ner_model = os.path.join(model_dir, 'ner.model')
parser_model = os.path.join(model_dir, 'parser.model')


if __name__ == '__main__':

    segmentor = Segmentor()
    segmentor.load_with_lexicon(cws_model, cwd_dict)
    # segmentor.load(cws_model)
    postagger = Postagger()
    postagger.load(pos_model)
    ner = NamedEntityRecognizer()
    ner.load(ner_model)
    parser = Parser()
    parser.load(parser_model)

    sentence_list = ['新加坡《联合早报》曝出了赵薇最近与上海知名人士汪雨的儿子汪道涵热恋。']
    for sentence in sentence_list:
        word_list = segmentor.segment(sentence)
        postags = postagger.postag(word_list)
        arcs = parser.parse(word_list, postags)
        nertags = ner.recognize(word_list, postags)

        person_entity_list = list(get_person_entity_set(word_list, nertags))
        dependency_tree = arcs_to_dependency_tree(arcs)

        for i in range(len(person_entity_list)):
            for j in range(i+1, len(person_entity_list)):
                q_set = [person_entity_list[i]]
                f_set = [person_entity_list[j]]

                modifier_set = get_modifier_set(word_list, dependency_tree, q_set[0], f_set[0])

                q_pi_vector = page_rank(q_set, word_list, dependency_tree)
                f_pi_vector = page_rank(f_set, word_list, dependency_tree)
                i_vector = get_I_vector(q_pi_vector, f_pi_vector)

                relation_word = get_relation_word_top_n(word_list, i_vector, postags, q_set, f_set, n=3)
                print(q_set[0], f_set[0], relation_word)
