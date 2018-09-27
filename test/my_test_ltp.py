from myfuncs import get_content_from_ltp
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
    sentence = '新加坡《联合早报》曝出了赵薇与上海知名人士汪雨的儿子汪道涵热恋。'
    word_list = segmentor.segment(sentence)
    content = get_content_from_ltp(' '.join(list(word_list)), 'sdp')
    print(content)