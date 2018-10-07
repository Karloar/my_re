from myfuncs import get_resource_path
from myfuncs import get_I_vector_by_qfset
from myfuncs import get_trigger_candidate
from myfuncs import get_qfset
from myfuncs import get_trigger_by_ap_cluster
from myfuncs import load_data_en
from myfuncs import print_running_time
from stanfordcorenlp import StanfordCoreNLP
from sklearn.cluster import AffinityPropagation


data_file = get_resource_path('data/TRAIN_FILE.TXT')


@print_running_time
def main():
    sentences, entity_relation_list = load_data_en(data_file)
    relation_trigger_dict = dict()

    with StanfordCoreNLP('http://localhost', port=9000, lang='en') as nlp:
        for sent, entity_relation in zip(sentences, entity_relation_list):
            word_list = nlp.word_tokenize(sent)
            dependency_tree = nlp.dependency_parse(sent)
            postags = [x[1] for x in nlp.pos_tag(sent)]
            q1_set = get_qfset(entity_relation[0], word_list)
            q2_set = get_qfset(entity_relation[2], word_list)
            i_vector = get_I_vector_by_qfset(q1_set, q2_set, word_list, dependency_tree)
            ap = AffinityPropagation().fit(i_vector)
            labels = ap.labels_
            trigger_candidate = get_trigger_candidate(word_list, i_vector, postags, q1_set, q2_set, style='stanfordcorenlp')
            trigger = get_trigger_by_ap_cluster(trigger_candidate, i_vector)
            relation_set = relation_trigger_dict.get(trigger[0], set())
            relation_set.add(entity_relation[1])
            relation_trigger_dict[trigger[0]] = relation_set
    print('-------------------------')
    for k, v in relation_trigger_dict.items():
        print(k, '-----', v)


if __name__ == '__main__':
    main()
    