from stanfordcorenlp import StanfordCoreNLP
import os
import logging
import numpy as np
import platform
from wl import get_sorted_word_I_list
from wl import get_trigger_candidate


cwd = os.getcwd()
mac_path = "/Users/karloar/Documents/other"
stanfordcorepath = os.path.join(mac_path, 'stanford-corenlp-full-2018-02-27')
if platform.system() == 'Windows':
    stanfordcorepath = os.path.join('I:\\python\\stanford', 'stanford-corenlp-full-2018-02-27')
data_file = os.path.join(cwd, 'train.txt')


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
        # print(neighbour)
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
        if satisfy_error(old_pi_vector, pi_vector, error):
            break
        old_pi_vector = pi_vector
        max_iter -= 1
    return pi_vector


def satisfy_error(old_pi_vector, pi_vector, error):
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


if __name__ == '__main__':
    # sentence_list, entity_relation = load_data(data_file)
    # sentence_list = ['新加坡《联合早报》曝出了赵薇与上海知名人士张三的儿子汪道涵热恋。']
    sentence_list = ['The surgeon cuts a small hole in the skull and lifts the edge of the brain to expose the nerve.']
    with StanfordCoreNLP(stanfordcorepath, lang='en', logging_level=None) as stanford:
        for sentence in sentence_list:
            dependency_tree = stanford.dependency_parse(sentence)
            postags_tuple = stanford.pos_tag(sentence)
            postags = [x[1] for x in postags_tuple]
            word_list = stanford.word_tokenize(sentence)
            print(word_list)
            print(postags)
            print(dependency_tree)
            # ner = stanford.ner(sentence)
            # print(ner)
            q_set = ['surgeon']
            f_set = ['hole']
            pr_vector = get_PR_vector(q_set, word_list)
            a_matrix = get_A_Matrix(word_list, dependency_tree)
            q_pi_vector = page_rank(q_set, word_list, dependency_tree)
            f_pi_vecgor = page_rank(f_set, word_list, dependency_tree)
            i_vector = get_I_vector(q_pi_vector, f_pi_vecgor)
            relation_trigger = get_trigger_candidate(word_list, i_vector, postags, q_set, f_set, lang='en')
            print(relation_trigger)
            for word, idx in relation_trigger:
                print(word, idx)
            # for word, i, _ in get_sorted_word_I_list(word_list, i_vector):
            #     print(word, i)
            

