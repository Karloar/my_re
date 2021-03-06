import numpy as np
from functools import cmp_to_key
import urllib
import urllib.request as request
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer
import os
from sklearn.cluster import AffinityPropagation
import re
from time import time
from stanfordcorenlp import StanfordCoreNLP
import pickle as pkl
import tensorflow as tf
import string
from attention import attention
from sklearn import metrics


def load_data_zh(file):
    sentence_list = []
    entity_relation = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            x = line.split()
            if len(x) == 4:
                sentence_list.append(x[-1])
                entity_relation.append((x[0], x[1], x[2]))
            elif len(x) == 1:
                sentence_list.append(x[-1])
    return sentence_list, entity_relation


def load_data_en(data_file, remove_other=False):
    '''
    加载英文数据, 返回句子列表以及(实体1, 关系, 实体2)列表。
    :return sent_list, entity_relation_list
    '''
    sent_list = []
    entity_relation_list = []
    with open(data_file, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            _, sent = lines[i].split('\t')
            e1 = re.findall(r'<e1>[\s\S]+</e1>', sent)[0][4:-5]
            e2 = re.findall(r'<e2>[\s\S]+</e2>', sent)[0][4:-5]
            sent = sent.replace('<e1>', ' ')
            sent = sent.replace('</e1>', ' ')
            sent = sent.replace('<e2>', ' ')
            sent = sent.replace('</e2>', ' ')
            relation = lines[i+1].strip()
            if len(re.findall(r'(\(e1,e2\)|\(e2,e1\))', relation)) > 0:
                relation = relation[:-7]
            # 去掉关系为Other的句子
            if remove_other and relation.lower() == 'other':
                continue
            sent_list.append(sent.strip()[1:-1])
            entity_relation_list.append((e1, relation, e2))
    return sent_list, entity_relation_list


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
        if i in r_set:
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


def page_rank(r_set, word_list, dependency_tree, max_iter=5000, error=1e-5, beta=0.3):
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


def get_I_vector_by_qfset(q_set, f_set, word_list, dependency_tree, max_iter=5000, error=1e-5, beta=0.3):
    '''
    根据q_set和f_set直接计算I(v|{Q, F})
    :param  q_set
    :param  f_set
    :param  word_list
    :param  max_iter = 5000
    :param  error = 1e-5
    :param  beta = 0.3
    '''
    q_i_vector = page_rank(q_set, word_list, dependency_tree, max_iter, error, beta)
    f_i_vector = page_rank(f_set, word_list, dependency_tree, max_iter, error, beta)
    return get_I_vector(q_i_vector, f_i_vector)



def get_sorted_word_I_list(word_list, i_vector, postags):
    '''
    将词语与对应的I值从大到小排序，返回(词语，I值)列表
    :param  word_list   分词后的词列表
    :param  i_vector    计算出的I向量
    :return (词语, I值, 索引)列表
    '''
    if type(i_vector) == np.ndarray:
        i_vector = i_vector.reshape((len(i_vector, ))).tolist()
    sorted_list = []
    for word, i, idx, postag in zip(word_list, i_vector, range(len(word_list)), postags):
        sorted_list.append((word, i, idx, postag))

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


def get_trigger_candidate(word_list, i_vector, postags, q_set, f_set, style='pyltp'):
    '''
    返回代表关系trigger的候选词语
    :param  word_list   分词后的词列表
    :param  i_vector    计算出的I 向量
    :param  postags     词性列表
    :param  q_set
    :param  f_set
    :param  style       词性标注的工具，默认是pyltp，可选：stanfordcorenlp
    :return relation_trigger_list  关系trigger词列表 (word, i_val, idx)
    '''
    sorted_word_i_list = get_sorted_word_I_list(word_list, i_vector, postags)
    pyltp_postag_set = {'N', 'V', 'A'}
    stanford_postag_set = {
        'NN', 'NNS', 'NNP', 'NNPS',
        'RB', 'RBR', 'RBS',
        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
        'JJ', 'JJR', 'JJS',
        'IN'
    }
    postag_set = pyltp_postag_set
    if style == 'stanfordcorenlp':
        postag_set = stanford_postag_set
    relation_trigger_list = list()
    for word, i_val, idx, postag in sorted_word_i_list:
        if idx in q_set or idx in f_set:
            continue
        if postags[idx].upper() not in postag_set:
            continue
        # 去掉标点
        if word[0] in string.punctuation:
            continue
        relation_trigger_list.append((word, i_val, idx, postag))
    return relation_trigger_list


def get_content_from_ltp(sentence, pattern):
    url_get_base = 'https://api.ltp-cloud.com/analysis/'
    args = {
        'api_key': '7221Y2o987TCDGsqSTjFccNNDtumGDVGCQANmJ6x',
        'text': sentence,
        'pattern': pattern,
        'format': 'plain'
    }
    result = request.urlopen(url_get_base, urllib.parse.urlencode(args).encode('utf-8'))
    content = result.read().decode('utf-8').strip()
    return content


# 这个函数需要重新写
def get_modifier_set(word_list, dependency_tree, entity_1, entity_2):
    '''
    得到实体1和实体2的修饰词集合
    :param  word_list   分词后的词列表
    :param  dependency_tree 句法已存关系树
    :param  entity_1    实体1
    :param  entity_2    实体2
    :return modifier_set    实体1和实体2的修饰词集合
    '''
    set_1 = _get_modifier_set_(word_list, dependency_tree, entity_1, entity_2)
    set_2 = _get_modifier_set_(word_list, dependency_tree, entity_2, entity_1)
    return set_1 | set_2


def dependency_tree_to_arr(dependency_tree, n):
    dependency_arr = []
    for i in range(n):
        dependency_arr.append([])
    for relation, from_idx, to_idx in dependency_tree:
        if from_idx != 0 and to_idx != 0:
            dependency_arr[from_idx - 1].append((to_idx - 1, relation))
    return dependency_arr


# 这个函数也需要重新写
def _get_modifier_set_(word_list, dependency_tree, entity_1, entity_2):
    '''
    查找实体1的修饰词，且实体2不能在修饰词中。
    :param  word_list   分词后的词列表
    :param  dependency_tree 句法已存关系树
    :param  entity_1    实体1
    :param  entity_2    实体2
    :return modifier_set    实体1的修饰词集合
    '''
    word_list = list(word_list)
    e_index_1 = word_list.index(entity_1)
    e_index_2 = word_list.index(entity_2)
    dependency_arr = dependency_tree_to_arr(dependency_tree, len(word_list))
    queue = []
    for x in dependency_arr[e_index_1]:
        if x[1].upper() == 'ATT':
            queue.append(x[0])
    modifier_set = set()
    while queue:
        x = queue[0]
        del queue[0]
        if x not in modifier_set:
            modifier_set.add(x)
            for a in dependency_arr[x]:
                if a[1].upper() == 'ATT':
                    queue.append(a[0])
    if e_index_2 in modifier_set and e_index_2 < e_index_1:
        temp = set()
        for x in modifier_set:
            if x >= e_index_2 and x < e_index_1:
                temp.add(x)
        modifier_set = modifier_set - temp
    return modifier_set


def init_pyltp(model_dir, dict_file=None):
    '''
    初始化Pyltp的几个模块
    :param  model_dir   模型的路径
    :param  dict_file   分词的外部词典
    :return segmentor, postagger, parser, ner
    '''
    segmentor = Segmentor()
    postagger = Postagger()
    parser = Parser()
    ner = NamedEntityRecognizer()
    
    cws_model = os.path.join(model_dir, 'cws.model')
    pos_model = os.path.join(model_dir, 'pos.model')
    parser_model = os.path.join(model_dir, 'parser.model')
    ner_model = os.path.join(model_dir, 'ner.model')

    if dict_file:
        segmentor.load_with_lexicon(cws_model, dict_file)
    else:
        segmentor.load(cws_model)
    postagger.load(pos_model)
    ner.load(ner_model)
    parser.load(parser_model)
    return segmentor, postagger, parser, ner


def get_trigger_candidate_vector(relation_trigger_list, word2vec_model):
    '''
    得到候选关系词语的词向量
    :param  relation_trigger_list 关系候选词语列表[(word, i_val, idx, postag)]
    :param  word2vec_model  训练好的word2vec模型
    :return 候选关系词语向量 n * 100
    '''
    word_vec = []
    for word, _, _, _ in relation_trigger_list:
        if word in word2vec_model:
            word_vec.append(word2vec_model[word])
        else:
            word_vec.append([0] * 100)
    return np.array(word_vec)


def get_trigger_by_ap_cluster(trigger_candidate, i_vector):
    '''
    通过AP聚类的结果得到关系trigger表示词语
    :param  trigger_candidate  关系候trigger选词语列表[(word, i_val, idx, postag)]
    :param  i_vector  I向量
    :return 关系trigger表示词语(word, i_val, idx)
    '''
    if len(trigger_candidate) == 1:
        return trigger_candidate[0]
    trigger_i_vector = [i_vector[idx] for _, _, idx, _ in trigger_candidate]
    ap = AffinityPropagation().fit(trigger_i_vector)
    cluster = dict()
    for (word, i_val, idx, postag), label in zip(trigger_candidate, ap.labels_):
        c = cluster.get(label, [])
        c.append((word, i_val, idx, postag))
        cluster[label] = c
    return _get_trigger_by_cluster_(cluster)


def _get_trigger_by_cluster_(cluster):
    cluster_i_val = dict()
    for c in cluster.keys():
        cluster_i_val[c] = (sum([x[1] for x in cluster[c]]) / len(cluster[c]))
    max_c_idx = list(cluster_i_val.keys())[0]
    max_val = cluster_i_val[max_c_idx]
    for c, val in cluster_i_val.items():
        if val > max_val:
            max_c_idx = c
    return max(cluster[max_c_idx], key=lambda x: x[1])


def get_resource_path(file):
    f = os.path.dirname(__file__)
    p = os.path.join(f, file)
    while not os.path.exists(p):
        f_new = os.path.dirname(f)
        if f_new == f:
            raise Exception("file not exist!")
        p = os.path.join(f_new, file)
        f = f_new
    return p


def get_indexes(element, collection):
    '''
    返回集合中所有元素的索引值
    '''
    return [idx for idx, x in zip(range(len(collection)), collection) if x == element]


def get_qfset(entity, word_list):
    '''
    根据实体以及词列表，返回qfset
    '''
    qf_set = []
    for x in entity.split():
        qf_set.extend(get_indexes(x, word_list))
    return qf_set


def print_running_time(*args, **kargs):
    '''
    在程序运行结束后显示运行时间，用@print_running_time修饰在函数上。
    可选参数：
    :param  show_func_name  True / False
    :param  message     自定义输出信息，在需要输出时间的地方用{:f},
    '''
    if len(args) == 1 and len(kargs) == 0:
        def _func(*fcargs, **fckargs):
            tic = time()
            func = args[0]
            return_val = func(*fcargs, **fckargs)
            toc = time()
            print("程序运行时间：{:f} 秒。".format(toc - tic))
            return return_val
        return _func

    if len(args) == 0 and len(kargs) != 0:
        def _func(func):
            def __func(*fcargs, **fckargs):
                tic = time()
                return_val = func(*fcargs, **fckargs)
                toc = time()
                func_name = '程序'
                if "message" in kargs:
                    print(kargs["message"].format(toc - tic))
                else:
                    if 'show_func_name' in kargs and kargs['show_func_name']:
                        func_name = "函数 " + func.__name__ + " "
                    print("{:s}运行时间：{:f} 秒。".format(func_name, toc - tic))
                return return_val
            return __func
        return _func


class Param:
    '''
    该类用来设定参数
    '''
    pass


def get_trigger_list_from_sents(sents, entity_relation_list, save_file=None, params: Param=None):
    '''
    根据句子列表得到trigger词语集合，并根据需要保存成文件以免重复计算, 格式[(trigger, i_val, idx), (...)]
    :param  sents                   句子集合
    :param  entity_relation_list    实体关系集合(实体1, 关系, 实体2), 默认为None   
    :param  save_file               将最终得到的向量集合保存成文件, 以免重复计算     
    :param  params  设定参数
                    style               使用stanfordcorenlp或者pyltp, 默认stanfordcorenlp    
                    max_iter            PageRank的最大迭代次数, 默认5000
                    error               PageRank的精确度, 默认1e-5
                    beta                PageRank的参数beta, 默认0.3
                    trigger_neighbour   trigger周围的单词数（包括trigger), 默认3
                    pyltp_model         pyltp模型目录, 默认是None
                    user_dict           pyltp用于分词的用户词典, 默认是None
                    
    :return trigger_vector_list
    '''
    # 参数处理
    style = 'stanfordcorenlp'
    max_iter = 5000
    error = 1e-5
    beta = 0.3
    pyltp_model = None
    user_dict = None
    if hasattr(params, 'style'):
        style = params.style
    if hasattr(params, 'max_iter'):
        max_iter = params.max_iter
    if hasattr(params, 'error'):
        error = params.error
    if hasattr(params, 'beta'):
        beta = params.beta
    if hasattr(params, 'pyltp_model'):
        pyltp_model = params.pyltp_model
    if hasattr(params, 'user_dict'):
        user_dict = params.user_dict
    
    if save_file and os.path.exists(save_file):
        word_list_file = get_word_list_file(save_file)
        return pkl.load(open(save_file, 'rb')), pkl.load(open(word_list_file, 'rb'))
    else:
        # 得到trigger
        nlp_tool = get_nlp_tool(style, pyltp_model, user_dict)
        trigger_list = []
        word_lists = []
        for sent, entity_relation in zip(sents, entity_relation_list):
            word_list, postags, dependency_tree = words_postags_dependency_tree(
                sent,
                nlp_tool,
                style=style
            )
            # 句子中的第一个单词的首字母转换成小写
            if len(word_list[0]) > 1 and word_list[0].upper() != word_list[0]:
                word_list[0] = word_list[0].lower()
            q1_set = get_qfset(entity_relation[0], word_list)
            q2_set = get_qfset(entity_relation[2], word_list)
            # 通过PageRank计算I值
            i_vector = get_I_vector_by_qfset(
                q1_set,
                q2_set,
                word_list,
                dependency_tree,
                max_iter=max_iter,
                beta=beta,
                error=error
            )
            trigger_candidate = get_trigger_candidate(
                word_list,
                i_vector,
                postags,
                q1_set,
                q2_set,
                style
            )
            trigger = get_trigger_by_ap_cluster(trigger_candidate, i_vector)
            trigger_list.append(trigger)
            word_lists.append(word_list)
        if style == 'stanfordcorenlp':
            nlp_tool.close()
        if save_file:
            dirname = os.path.dirname(save_file)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            word_list_file = get_word_list_file(save_file)
            pkl.dump(trigger_list, open(save_file, 'wb'))
            pkl.dump(word_lists, open(word_list_file, 'wb'))
        return trigger_list, word_lists


def get_word_list_file(save_file):
    dirname = os.path.dirname(save_file)
    filename = 'word_list_' + os.path.basename(save_file)
    return os.path.join(dirname, filename)


def get_trigger_neighbour_vector_list(trigger_neighbour_list, model, vector_size):
    '''
    使用word2vec将trigger_neighbour_list中的每个单词转换成向量, 返回的是列表
    '''
    trigger_neighbour_vector_list = []
    for trigger_neighbour in trigger_neighbour_list:
        trigger_neighbour_vector = []
        for word in trigger_neighbour:
            if word in model:
                trigger_neighbour_vector.append(model[word])
            else:
                trigger_neighbour_vector.append(np.zeros((1, vector_size)))
        trigger_neighbour_vector_list.append(trigger_neighbour_vector)
    return trigger_neighbour_vector_list


def get_feature_vector_for_nn(trigger_neighbour_vector_list, vector_size):
    '''
    将trigger_neighbour_vector_list转换成用于神经网络训练的向量
    '''
    data_num = len(trigger_neighbour_vector_list)
    feature_matrix = np.zeros((data_num, vector_size))
    for i in range(data_num):
        for vector in trigger_neighbour_vector_list[i]:
            feature_matrix[i, :] += vector.ravel()
        feature_matrix[i, :] /= len(trigger_neighbour_vector_list[i])
    return feature_matrix.astype(np.float32)


def get_feature_vector_for_nn2(word_lists, trigger_list, vector_size, model, entity_relation_list):
    '''
    包含位置信息、词性信息的word2vec特征，用于神经网络训练，维数：vector_size
    '''
    sent_num = len(word_lists)
    feature_matrix = np.zeros((sent_num, vector_size))
    for i in range(sent_num):
        trigger, _, trigger_idx, postag = trigger_list[i]
        len_word_list = len(word_lists[i])
        feature_matrix[i, 0] = trigger_idx / len(word_lists[i])
        feature_matrix[i, 1] = (get_entity_value(entity_relation_list[i][0], word_lists[i]) - trigger_idx) / len_word_list
        feature_matrix[i, 2] = (get_entity_value(entity_relation_list[i][2], word_lists[i]) - trigger_idx) / len_word_list
        # feature_matrix[i, 3 = get_postag_value(postag)
        if trigger in model:
            feature_matrix[i, 3:] = model[trigger]
    return feature_matrix.astype(np.float32)


def get_feature_vector_for_classify(word_lists, trigger_list, entity_relation_list):
    sent_num = len(word_lists)
    feature_matrix = np.zeros((sent_num, 4))
    for i in range(sent_num):
        _, _, trigger_idx, postag = trigger_list[i]
        len_word_list = len(word_lists[i])
        feature_matrix[i, 0] = (trigger_idx + 1) / len_word_list
        feature_matrix[i, 1] = get_postag_value(postag)
        feature_matrix[i, 2] = (get_entity_value(entity_relation_list[i][0], word_lists[i]) + 1) / len_word_list
        feature_matrix[i, 3] = (get_entity_value(entity_relation_list[i][2], word_lists[i]) + 1) / len_word_list
    return feature_matrix


def get_feature_vector_for_rnn(trigger_neighbour_vector_list, vector_size):
    n_steps = len(trigger_neighbour_vector_list[0])
    data_num = len(trigger_neighbour_vector_list)
    feature_vector = np.zeros((data_num, n_steps * vector_size))
    for i in range(data_num):
        for j in range(n_steps):
            from_idx = j * vector_size
            to_idx = from_idx + vector_size
            feature_vector[i, from_idx:to_idx] = np.array(trigger_neighbour_vector_list[i][j]).reshape(1, vector_size)
    return feature_vector.astype(np.float32)


def get_feature_vector_for_rnn2(trigger_neighbour_vector_list, vector_size, word_lists, trigger_list, entity_relation_list, use_entity=False):
    n_steps = len(trigger_neighbour_vector_list[0])
    data_num = len(trigger_neighbour_vector_list)
    pre_vector_size = vector_size
    vector_size = 2 + vector_size
    if use_entity:
        vector_size = 4 + vector_size
    feature_vector = np.zeros((data_num, n_steps * vector_size))
    for i in range(data_num):
        _, _, trigger_idx, postag = trigger_list[i]
        len_word_list = len(word_lists[i])
        vector = np.zeros((1, vector_size))
        vector[0, 0] = (trigger_idx + 1) / len_word_list
        vector[0, 1] = get_postag_value(postag)
        if use_entity:
            feature_matrix[0, 2] = (get_entity_value(entity_relation_list[i][0], word_lists[i]) + 1) / len_word_list
            feature_matrix[0, 3] = (get_entity_value(entity_relation_list[i][2], word_lists[i]) + 1) / len_word_list
        for j in range(n_steps):
            from_idx = j * vector_size
            to_idx = from_idx + vector_size
            vector[0, vector_size-pre_vector_size:] = np.array(trigger_neighbour_vector_list[i][j]).reshape(1, pre_vector_size)
            feature_vector[i, from_idx:to_idx] = vector
    return feature_vector.astype(np.float32)


def get_feature_vector_for_rnn3(trigger_neighbour_vector_list, vector_size):
    n_steps = len(trigger_neighbour_vector_list[0])
    data_num = len(trigger_neighbour_vector_list)
    feature_vector = np.zeros((data_num, n_steps * vector_size))
    for i in range(data_num):
        for j in range(n_steps):
            from_idx = j * vector_size
            to_idx = from_idx + vector_size
            feature_vector[i, from_idx:to_idx] = np.array(trigger_neighbour_vector_list[i][j]).reshape(1, vector_size)
    return feature_vector.astype(np.float32).reshape((data_num, n_steps, -1))


# def add_postion_postag_feature(data, word_lists, trigger_list, entity_relation_list, use_entity=False):
#     '''

#     '''
#     ppf = get_feature_vector_for_classify(word_lists, trigger_list, entity_relation_list)
#     if use_entity:
#         return np.concatenate([data, ppf], axis=1)
#     return np.concatenate([data, ppf[:, :2]], axis=1)


def get_postag_value(postag):
    stanford_postags = [
        ['NN', 'NNS', 'NNP', 'NNPS'],
        ['RB', 'RBR', 'RBS'],
        ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        ['JJ', 'JJR', 'JJS'],
        ['IN']
    ]
    postag_num = len(stanford_postags)
    idx = 0
    for i in range(postag_num):
        if postag.upper() in stanford_postags[i]:
            idx = i + 1
            break
    return idx / postag_num


def get_entity_value(entity, word_list):
    idx_list = []
    for x in entity.split():
        idx_list.extend(get_indexes(x, word_list))
    return sum(idx_list) / len(idx_list)


def get_trigger_neighbour_words(trigger, word_list, trigger_neighbour, remove_punc=False):
    '''
        返回trigger附近的词，去除标点，如果句子的长度达不到要求，则用'@@@'来补全
    '''
    n = trigger_neighbour // 2
    word_idx_list = get_word_idx_list(word_list)
    len_word_idx_list = len(word_idx_list)
    trigger_idx = trigger[2]
    sent_len = len(word_list)
    if remove_punc:
        trigger_idx = word_idx_list.index(trigger[2])
        sent_len = len_word_idx_list
    if trigger_neighbour <= sent_len and trigger_neighbour > 0:
        if trigger_idx - n >= 0 and trigger_idx + n < sent_len:
            if remove_punc:
                rtv_word_list = word_idx_list[trigger_idx-n:trigger_idx+n+1]
            else:
                rtv_word_list = word_list[trigger_idx-n:trigger_idx+n+1]
        elif trigger_idx - n < 0:
            if remove_punc:
                rtv_word_list = word_idx_list[:trigger_neighbour]
            else:
                rtv_word_list = word_list[:trigger_neighbour]
        elif trigger_idx + n >= len_word_idx_list:
            if remove_punc:
                rtv_word_list = word_idx_list[sent_len-trigger_neighbour:]
            else:
                rtv_word_list = word_list[sent_len-trigger_neighbour:]
    elif trigger_neighbour > sent_len:
        diff_len = trigger_neighbour - sent_len
        rtv_word_list = word_list
        if remove_punc:
            rtv_word_list = list(map(lambda idx: word_list[idx], word_idx_list))
        rtv_word_list.extend(['@@@'] * diff_len)
        return rtv_word_list
    if remove_punc:
        return list(map(lambda idx: word_list[idx], rtv_word_list))
    return rtv_word_list


def get_trigger_neighbour_words2(trigger, word_list, trigger_neighbour, remove_punc=False):
    '''
    严格按照tigger在中间的形式，如果没有单词，则补"@@@".
    '''
    n = trigger_neighbour // 2
    word_idx_list = get_word_idx_list(word_list)
    len_word_idx_list = len(word_idx_list)
    trigger_idx = trigger[2]
    sent_len = len(word_list)
    if remove_punc:
        trigger_idx = word_idx_list.index(trigger[2])
        sent_len = len_word_idx_list
    if trigger_idx - n >= 0 and trigger_idx + n < sent_len:
        if remove_punc:
            rtv_word_list = map(lambda x: word_list[x], word_idx_list[trigger_idx-n:trigger_idx+n+1])
        else:
            rtv_word_list = word_list[trigger_idx-n:trigger_idx+n+1]
    elif trigger_idx + n < sent_len:
        diff = n - trigger_idx
        rtv_word_list = ['@@@'] * diff
        if remove_punc:
            rtv_word_list.extend(map(lambda x: word_list[x], word_idx_list[:trigger_idx+n+1]))
        else:
            rtv_word_list.extend(word_list[:trigger_idx+n+1])
    elif trigger_idx - n >= 0:
        diff = trigger_idx + n + 1 - sent_len
        if remove_punc:
            rtv_word_list = map(lambda x: word_list[x], word_idx_list[trigger_idx - n:])
        else:
            rtv_word_list = word_list[trigger_idx - n:]
        rtv_word_list.extend(['@@@'] * diff)
    else:
        left_diff = n - trigger_idx
        right_diff = trigger_idx + n + 1 - sent_len
        rtv_word_list = ['@@@'] * left_diff
        if remove_punc:
            rtv_word_list.extend(map(lambda x: word_list[x], word_idx_list))
        else:
            rtv_word_list.extend(word_list)
        rtv_word_list.extend(['@@@'] * right_diff)
    return rtv_word_list


def get_trigger_neighbour_words_list(trigger_list, word_lists, trigger_neighbour, remove_punc=True, type=1):
    trigger_neighbour_word_list = []
    for trigger, word_list in zip(trigger_list, word_lists):
        if type == 1:
            trigger_neighbour_word_list.append(get_trigger_neighbour_words(
                trigger,
                word_list,
                trigger_neighbour,
                remove_punc
            ))
        elif type == 2:
            trigger_neighbour_word_list.append(get_trigger_neighbour_words2(
                trigger,
                word_list,
                trigger_neighbour,
                remove_punc
            ))
    return trigger_neighbour_word_list


def get_word_idx_list(word_list):
    idx_list = []
    for i in range(len(word_list)):
        if not word_list[i][0] in string.punctuation:
            idx_list.append(i)
    return idx_list


def get_nlp_tool(style, pyltp_model=None, user_dict=None):
    if style == 'pyltp':
        if not pyltp_model:
            raise Exception
        segmentor, postagger, parser, _ = init_pyltp(pyltp_model, user_dict)
        return (segmentor, postagger, parser)
    else:
        return StanfordCoreNLP('http://localhost', port=9000, lang='en')


def words_postags_dependency_tree(sent, nlp_tool, style):
    if style == 'pyltp':
        segmentor, postagger, parser = nlp_tool
        word_list = list(segmentor.segment(sent))
        postags = list(postagger.postag(word_list))
        arcs = parser.parse(word_list, postags)
        dependency_tree = arcs_to_dependency_tree(arcs)
    else:
        word_list = nlp_tool.word_tokenize(sent)
        dependency_tree = nlp_tool.dependency_parse(sent)
        postags = [x[1] for x in nlp_tool.pos_tag(sent)]
    return word_list, postags, dependency_tree


def get_batch(data, batch_size, step):
    '''
    根据迭代的步骤得到数据batch
    '''
    data_num = len(data)
    batch_num = int(data_num / batch_size + 0.5)
    from_idx = step % batch_num * batch_size
    to_idx = from_idx + batch_size
    choose_idx = list(range(data_num))
    choose_idx.extend(list(range(data_num)))
    return data[choose_idx[from_idx:to_idx], :]


def get_label_by_entity_relation_list(entity_relation_list: list, relation_list: list):
    '''
    根据类别将类别映射成向量
    '''
    m = len(entity_relation_list)
    n = len(relation_list)
    label = np.zeros((m, n)).astype(np.float32)
    for i in range(m):
        idx = relation_list.index(entity_relation_list[i][1])
        label[i, idx] = 1
    return label


class MyRNNClassifier:

    def __init__(
        self,
        n_inputs,
        n_classes,
        n_steps,
        n_hidden_units,
        keep_prob,
        learing_rate=0.001,
        batch_size=200,
        training_iter=2000,
        use_bilstm=False,
        use_attention=False
    ):
        self._nsteps = n_steps
        self._learning_rate = learing_rate
        self._batch_size = batch_size
        self._nclasses = n_classes
        self._nhidden_units = n_hidden_units
        self._ninputs = n_inputs
        self._training_iter = training_iter
        self._keep_prob = keep_prob
        self._use_attention = use_attention
        self._use_bilstm = use_bilstm
        self.create_graph()
    
    def create_graph(self):
        self._x = tf.placeholder(tf.float32, [None, self._nsteps, self._ninputs])
        self._y = tf.placeholder(tf.float32, [None, self._nclasses])
        self._nbatch_size = tf.placeholder(tf.int32, [], name="batch_size")
        self._weights = {
            'in': tf.Variable(tf.random_normal([self._ninputs, self._nhidden_units]) * 0.01),
            'out': tf.Variable(tf.random_normal([self._nhidden_units, self._nclasses]) * 0.01)
        }
        if self._use_bilstm:
            self._weights = {
                'in': tf.Variable(tf.random_normal([self._ninputs, self._nhidden_units]) * 0.01),
                'out': tf.Variable(tf.random_normal([self._nhidden_units * 2, self._nclasses]) * 0.01)
            }
        self._biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[self._nhidden_units, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[self._nclasses, ]))
        }

        self._pred = self.rnn(self._x, self._weights, self._biases)
        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._pred, labels=self._y))
        self._train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self._cost)
        self._correct_pred = tf.equal(tf.argmax(self._pred, 1), tf.argmax(self._y, 1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))
    
    def rnn(self, X, weight, biase):
        X = tf.reshape(X, [-1, self._ninputs])
        X_in = tf.matmul(X, weight['in']) + biase['in']
        X_in = tf.reshape(X_in, [-1, self._nsteps, self._nhidden_units])
        if self._use_bilstm:
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self._nhidden_units, forget_bias=1, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self._nhidden_units, forget_bias=1, state_is_tuple=True)
            _init_state_fw = cell_fw.zero_state(self._nbatch_size, dtype=tf.float32)
            _init_state_bw = cell_bw.zero_state(self._nbatch_size, dtype=tf.float32)
            (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                X_in,
                initial_state_fw=_init_state_fw,
                initial_state_bw=_init_state_bw,
                time_major=False
            )
            outputs = tf.concat([output_fw, output_bw], axis=-1)
        else:
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self._nhidden_units, forget_bias=1, state_is_tuple=True)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self._keep_prob)
            _init_state = lstm_cell.zero_state(self._nbatch_size, dtype=tf.float32)
            outputs, state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
        
        value = outputs[:, -1, :]
        if self._use_attention:
            value = attention(outputs, self._nhidden_units)
        results = tf.matmul(value, weight['out']) + biase['out']
        return results

    def fit(self, data, label):
        self._sess = tf.Session()
        init = tf.global_variables_initializer()
        self._sess.run(init)

        for step in range(self._training_iter):
            batch_xs = get_batch(data, self._batch_size, step)
            # # 确保batch_size与数据量一致
            # self._batch_size = min([self._batch_size, len(batch_xs)])

            batch_xs = batch_xs.reshape([self._batch_size, self._nsteps, self._ninputs])
            batch_ys = get_batch(label, self._batch_size, step)
            loss, _ = self._sess.run([self._cost, self._train_op], feed_dict={
                self._nbatch_size: self._batch_size,
                self._x: batch_xs,
                self._y: batch_ys
            })
            # if step % 10 == 0:
            #     print(step, '--', self._sess.run(self._accuracy, feed_dict={
            #         self._nbatch_size: self._batch_size,
            #         self._x: batch_xs,
            #         self._y: batch_ys
            #     }))
            # print(step, '  loss:', loss)
    
    def accuracy(self, data, label):
        data = data.reshape([data.shape[0], self._nsteps, self._ninputs])
        return self._sess.run(self._accuracy, feed_dict={
            self._nbatch_size: data.shape[0],
            self._x: data,
            self._y: label
        })
    
    def report(self, data, label):
        data = data.reshape([data.shape[0], self._nsteps, self._ninputs])
        pred = self._sess.run(self._pred, feed_dict={
            self._nbatch_size: data.shape[0],
            self._x: data,
            self._y: label
        })
        return metrics.classification_report(np.argmax(label, axis=1), np.argmax(pred, axis=1))
        

    def close(self):
        self._sess.close()
    # def __enter__(self):
    #     return self

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     self.sess.close()
