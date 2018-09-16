import numpy as np
from functools import cmp_to_key


def load_data(file):
    sentence_list = []
    entity_relation = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            x = line.split()
            sentence_list.append(x[-1])
            entity_relation.append((x[0], x[1], x[2]))
    return sentence_list, entity_relation


def get_PR_vector(r_set, word_list):
    '''
        get PR Vector: n by 1
        :param r_set
        :param word_list
        :return PR Vector
    '''
    word_list_len = len(word_list)
    r_set_len = len(r_set)
    pr = np.zeros((word_list_len, 1))
    for i in range(word_list_len):
        if word_list[i] in r_set:
            pr[i, 0] = 1 / r_set_len
    return pr


def get_A_Matrix(word_list, dependency_tree):
    '''
        get A Matrix: n by n
        :param  word_list
        :param  dependency_tree
        :return  A Matrix
    '''
    word_list_len = len(word_list)
    a_matrix = np.zeros((word_list_len, word_list_len))
    for i in range(word_list_len):
        neighbour = get_neighbour(word_list[i], i, dependency_tree)
        neighbour_num = len(neighbour)
        for x in neighbour:
            a_matrix[x][i] = 1 / neighbour_num
    return a_matrix


def get_neighbour(word, idx, dependency_tree):
    '''
        Get the neighbours of the word in dependency tree
        :param  word
        :param  idx: the index of word in word_list
        :param  dependency_tree
    '''
    neighbour = set()
    for _, from_idx, to_idx in dependency_tree:
        if from_idx - 1 == idx:
            if to_idx != 0:
                neighbour.add(to_idx - 1)
        if to_idx - 1 == idx:
            if from_idx != 0:
                neighbour.add(from_idx - 1)
    return neighbour


def get_from_neighbour(word, idx, dependency_tree):
    '''
        找到指向当前词的所有词语
        :param  word
        :param  idx: the index of word in word_list
        :param  dependency_tree
    '''
    from_neighbour = set()
    for _, from_idx, to_idx in dependency_tree:
        if to_idx - 1 == idx and from_idx != 0:
            from_neighbour.add(from_idx - 1)
    return from_neighbour


def get_to_neighbour(word, idx, dependency_tree):
    '''
        找到当前词指向的所有词语
        :param  word
        :param  idx: the index of word in word_list
        :param  dependency_tree
    '''
    to_neighbour = set()
    for _, from_idx, to_idx in dependency_tree:
        if from_idx - 1 == idx and to_idx != 0:
            to_neighbour.add(to_idx - 1)
    return to_neighbour


def arcs_to_dependency_tree(arcs):
    dependency_tree = []
    for i, arc in zip(range(len(arcs)), arcs):
        dependency_tree.append((arc.relation, arc.head, i + 1))
    return dependency_tree


def page_rank(r_set, word_list, dependency_tree, max_iter=1000, error=1e-5, beta=0.3):
    '''
        Calculate I(v|R), Get PI Vector by Page Rank
        :param  r_set
        :param  word_list
        :param  dependency_tree
        :param  max_iter
        :param  error
        :param  beta
        :return PI Vector   I(v|R) = pi(v)
    '''
    a_matrix = get_A_Matrix(word_list, dependency_tree)
    pr_vector = get_PR_vector(r_set, word_list)
    word_list_num = len(word_list)
    old_pi_vector = np.ones((word_list_num, 1)) * (1 / word_list_num)
    while max_iter > 0:
        pi_vector = cal_pi_vector(old_pi_vector, a_matrix, pr_vector, beta)
        if _satisfy_error(old_pi_vector, pi_vector, error):
            break
        old_pi_vector = pi_vector
        max_iter -= 1
    return pi_vector


def _satisfy_error(old_pi_vector, pi_vector, error):
    vec = np.abs(old_pi_vector - pi_vector)
    if len(np.where(vec >= error)[0]) > 0:
        return False
    return True


def cal_pi_vector(pi_vector, a_matrix, pr_vector, beta):
    '''
        calculate PI Vector
    '''
    pi_vector = (1 - beta) * np.dot(a_matrix, pi_vector) + beta * pr_vector
    return pi_vector


def get_I_vector(q_pi_vector, f_pi_vector):
    '''
        Calculate I(v|{Q, F})
        :param  q_pi_vector I(v|Q)
        :param  f_pi_vector I(v|F)
    '''
    return f_pi_vector + f_pi_vector * q_pi_vector + q_pi_vector


def get_sorted_word_I_list(word_list, i_vector):
    '''
        将词语与对应的I值从大到小排序，返回(词语，I值)列表
        :param  word_list   分词后的词列表
        :param  i_vector    计算出的I向量
        :return (词语，I值)列表
    '''
    if type(i_vector) == np.ndarray:
        i_vector = i_vector.reshape((len(i_vector, ))).tolist()
    sorted_list = []
    for word, i, idx in zip(word_list, i_vector, range(len(word_list))):
        sorted_list.append((word, i, idx))

    def mycmp(x, y):
        if x[1] > y[1]:
            return -1
        if x[1] < y[1]:
            return 1
        return 0

    sorted_list = sorted(sorted_list, key=cmp_to_key(mycmp))
    return sorted_list


def get_person_entity_set(word_list, nertags):
    '''
        从分词后的词列表中找到人物实体
        :param  word_list   分词后的词列表
        :param  nertags     标记列表
        :return 人物实体集合
    '''
    person_entity = set()
    for word, nertag in zip(word_list, nertags):
        if nertag == 'S-Nh':
            person_entity.add(word)
    return person_entity


def get_relation_word(word_list, i_vector, postags, q_set, r_set):
    sorted_word_i_list = get_sorted_word_I_list(word_list, i_vector)
    for word, i, idx in sorted_word_i_list:
        if word in q_set or word in r_set:
            continue
        
    
