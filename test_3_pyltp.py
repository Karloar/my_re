from myfuncs import page_rank, get_I_vector
from myfuncs import arcs_to_dependency_tree, get_sorted_word_I_list
from myfuncs import get_person_entity_set
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

    sentence_list = ['新加坡《联合早报》曝出了赵薇与上海知名人士张三的儿子汪道涵热恋中。']
    q_set = ['赵薇']
    f_set = ['汪道涵']

    word_list = segmentor.segment(sentence_list[0])
    postags = postagger.postag(word_list)
    arcs = parser.parse(word_list, postags)
    dependency_tree = arcs_to_dependency_tree(arcs)
    nertags = ner.recognize(word_list, postags)
    person_entity_set = get_person_entity_set(word_list, nertags)

    q_pi_vector = page_rank(q_set, word_list, dependency_tree)
    f_pi_vector = page_rank(f_set, word_list, dependency_tree)

    i_vector = get_I_vector(q_pi_vector, f_pi_vector)
    
    sorted_word_I_list = get_sorted_word_I_list(word_list, i_vector)
    for word, i_value, _ in sorted_word_I_list:
        print(word, i_value)

