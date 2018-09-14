from myfuncs import page_rank, get_I_vector
from myfuncs import arcs_to_dependency_tree
from pyltp import Segmentor, NamedEntityRecognizer, Parser, Postagger
import os
import numpy as np


cwd = os.getcwd()
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

    sentence_list = ['新加坡《联合早报》曝出了赵薇与上海知名人士汪雨的儿子汪道涵热恋中。']
    q_set = ['赵薇']
    f_set = ['汪道涵']

    word_list = segmentor.segment(sentence_list[0])
    postags = postagger.postag(word_list)
    arcs = parser.parse(word_list, postags)
    dependency_tree = arcs_to_dependency_tree(arcs)
    nertags = ner.recognize(word_list, postags)
    for word, postag in zip(word_list, postags):
        print(word, postag)

    # print(dependency_tree)

    q_pi_vector = page_rank(q_set, word_list, dependency_tree)
    f_pi_vector = page_rank(f_set, word_list, dependency_tree)

    i_vector = get_I_vector(q_pi_vector, f_pi_vector)
    
    word_vector = np.array(word_list)
    sorted_word_vector = word_vector[np.argsort(i_vector.T)]
    sorted_i_vector = np.sort(i_vector.T)

    for w, i in zip(sorted_word_vector[0], sorted_i_vector[0]):
        print(w, i)


