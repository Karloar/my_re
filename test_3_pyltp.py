from myfuncs import page_rank, get_I_vector, get_trigger_candidate
from myfuncs import arcs_to_dependency_tree, init_pyltp, get_resource_path
from myfuncs import get_person_entity_set, get_modifier_set
from myfuncs import get_trigger_candidate_vector, get_trigger_by_ap_cluster
from myfuncs import get_qfset
from myfuncs import print_running_time
import os
import platform
from gensim.models import Word2Vec
from sklearn.cluster import AffinityPropagation

model_dir = '/Users/karloar/Documents/other/ltp_data_v3.4.0'
if platform.system() == 'Windows':
    model_dir = r'E:\ltp_data'

dict_file = get_resource_path('data/dict.txt')
wiki_model_file = os.path.join(model_dir, 'wiki_model')


@print_running_time
def main():
    segmentor, postagger, parser, ner = init_pyltp(model_dir, dict_file)
    wiki_model = Word2Vec.load(wiki_model_file)
    sentence_list = [
        '新加坡《联合早报》曝出了赵薇最近与上海知名人士汪雨的儿子汪道涵热恋。',
        '从相恋到结婚，辛柏青和朱媛媛的爱情生活也是鲜有人知。',
        '廖国栋呼吁陈水扁放开心胸。',
        '张歆艺与好友蒋欣在微博上传的搞笑文章笑翻网友',
        '吴奇隆与刘诗诗去年领证，婚礼筹备了1年'
    ]
    for sentence in sentence_list:
        word_list = list(segmentor.segment(sentence))
        postags = postagger.postag(word_list)
        arcs = parser.parse(word_list, postags)
        nertags = ner.recognize(word_list, postags)
     
        person_entity_list = list(get_person_entity_set(word_list, nertags))
        dependency_tree = arcs_to_dependency_tree(arcs)

        for i in range(len(person_entity_list)):
            for j in range(i+1, len(person_entity_list)):
                q_set = get_qfset(person_entity_list[i], word_list)
                f_set = get_qfset(person_entity_list[j], word_list)

                modifier_set = get_modifier_set(word_list, dependency_tree, word_list[q_set[0]], word_list[f_set[0]])

                q_pi_vector = page_rank(q_set, word_list, dependency_tree)
                f_pi_vector = page_rank(f_set, word_list, dependency_tree)
                i_vector = get_I_vector(q_pi_vector, f_pi_vector)

                trigger_candidate_list = get_trigger_candidate(word_list, i_vector, postags, q_set, f_set)
                # trigger_candidate_vector = get_trigger_candidate_vector(trigger_candidate_list, wiki_model)
                print(word_list[q_set[0]], word_list[f_set[0]])
                trigger = get_trigger_by_ap_cluster(trigger_candidate_list, i_vector)
                print(trigger)
                # print(trigger_candidate_vector)
 

if __name__ == '__main__':

    main()
