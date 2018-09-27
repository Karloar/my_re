import urllib.request as request
import urllib
import pickle as pkl
import os
from myfuncs import page_rank, get_I_vector
from myfuncs import get_sorted_word_I_list
from myfuncs import get_person_entity_set
from myfuncs import get_content_from_ltp
from pyltp import Segmentor, NamedEntityRecognizer, Parser, Postagger
import platform


cwd = os.getcwd()
test_ltp = os.path.join(cwd, 'test_ltp.pkl')
model_dir = '/Users/karloar/Documents/other/ltp_data_v3.4.0'
if platform.system() == 'Windows':
    model_dir = r'E:\ltp_data'

cws_model = os.path.join(model_dir, 'cws.model')
cwd_dict = os.path.join(cwd, 'dict.txt')
pos_model = os.path.join(model_dir, 'pos.model')
ner_model = os.path.join(model_dir, 'ner.model')
parser_model = os.path.join(model_dir, 'parser.model')


def get_sematic_dependency_by_content(content):
    dependency_tree = []
    for line in content.split('\n'):
        x = line.split(' ')
        if '_' in x[0]:
            to_idx = int(x[0][x[0].index('_', True) + 1:])
        else:
            to_idx = int(x[0])
        if '_' in x[1]:
            from_idx = int(x[1][x[1].index('_', True) + 1:])
        else:
            from_idx = int(x[1])
        dependency_tree.append((x[2], from_idx + 1, to_idx + 1))
    return dependency_tree


if __name__ == '__main__':
    sentence_list = ['新加坡《联合早报》曝出了赵薇与上海知名人士张三的儿子汪道涵热恋。']
    if os.path.exists(test_ltp):
        content = pkl.load(open(test_ltp, 'rb'))
    else:
        content = get_content_from_ltp(sentence_list[0], 'sdp')
        pkl.dump(content, open(test_ltp, 'wb'))
    dependency_tree = get_sematic_dependency_by_content(content)
    segmentor = Segmentor()
    # segmentor.load_with_lexicon(cws_model, cwd_dict)
    segmentor.load(cws_model)
    postagger = Postagger()
    postagger.load(pos_model)
    ner = NamedEntityRecognizer()
    ner.load(ner_model)
    parser = Parser()
    parser.load(parser_model)

    q_set = ['赵薇']
    f_set = ['汪道涵']

    word_list = segmentor.segment(sentence_list[0])
    print(list(word_list))
    print(len(word_list))

    postags = postagger.postag(word_list)
    arcs = parser.parse(word_list, postags)

    q_pi_vector = page_rank(q_set, word_list, dependency_tree)
    f_pi_vector = page_rank(f_set, word_list, dependency_tree)

    i_vector = get_I_vector(q_pi_vector, f_pi_vector)
    
    sorted_word_I_list = get_sorted_word_I_list(word_list, i_vector)
    for word, i_value, _ in sorted_word_I_list:
        print(word, i_value)