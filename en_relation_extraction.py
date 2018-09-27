from wl import get_resource_path
import re


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
            sent_list.append(sent)
            entity_relation_list.append((e1, relation, e2))
    return sent_list, entity_relation_list


if __name__ == '__main__':
    sentences, entity_relation_list = load_data(data_file)
