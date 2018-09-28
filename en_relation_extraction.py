from myfuncs import get_resource_path
from myfuncs import get_I_vector_by_qfset
from myfuncs import get_trigger_candidate
from myfuncs import get_qfset
from myfuncs import get_trigger_by_ap_cluster
import re
from stanfordcorenlp import StanfordCoreNLP
from sklearn.cluster import AffinityPropagation


data_file = get_resource_path('data/TRAIN_FILE.TXT')


def load_data(data_file):
    sent_list = []
    entity_relation_list = []
    with open(data_file, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            _, sent = lines[i].split('\t')
            e1 = re.findall(r'<e1>[\s\S]+</e1>', sent)[0][4:-5]
            e2 = re.findall(r'<e2>[\s\S]+</e2>', sent)[0][4:-5]
            sent = sent.replace('<e1>', '')
            sent = sent.replace('</e1>', '')
            sent = sent.replace('<e2>', '')
            sent = sent.replace('</e2>', '')
            relation = lines[i+1].strip()
            sent_list.append(sent.strip()[1:-1])
            entity_relation_list.append((e1, relation, e2))
    return sent_list, entity_relation_list


if __name__ == '__main__':
    sentences, entity_relation_list = load_data(data_file)
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
            print(trigger)
