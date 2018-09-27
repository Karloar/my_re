from stanfordcorenlp import StanfordCoreNLP
from wl.myfuncs import page_rank, get_I_vector
from wl import get_sorted_word_I_list
from wl import get_person_entity_set, get_modifier_set


if __name__ == '__main__':
    nlp = StanfordCoreNLP('http://localhost', port=9000, lang='zh')
    sentence = '新加坡《联合早报》曝出了赵薇与上海知名人士张三的儿子汪道涵热恋。'
    word_list = nlp.word_tokenize(sentence)
    dependency_tree = nlp.dependency_parse(sentence)
    q_set = ['赵薇']
    f_set = ['汪道涵']
    q_pi_vector = page_rank(q_set, word_list, dependency_tree)
    f_pi_vecgor = page_rank(f_set, word_list, dependency_tree)
    i_vector = get_I_vector(q_pi_vector, f_pi_vecgor)

    print(word_list)
    print('-------------------------------')
    for x in dependency_tree:
        print(x)
    print('-------------------------------')

    for word, i, _ in get_sorted_word_I_list(word_list, i_vector):
        print(word, i)