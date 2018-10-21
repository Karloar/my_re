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


def load_data_en(data_file):
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
            sent = sent.replace('<e1>', '')
            sent = sent.replace('</e1>', '')
            sent = sent.replace('<e2>', '')
            sent = sent.replace('</e2>', '')
            relation = lines[i+1].strip()
            if len(re.findall(r'(\(e1,e2\)|\(e2,e1\))', relation)) > 0:
                relation = relation[:-7]
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



def get_sorted_word_I_list(word_list, i_vector):
    '''
    将词语与对应的I值从大到小排序，返回(词语，I值)列表
    :param  word_list   分词后的词列表
    :param  i_vector    计算出的I向量
    :return (词语, I值, 索引)列表
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
    sorted_word_i_list = get_sorted_word_I_list(word_list, i_vector)
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
    for word, i_val, idx in sorted_word_i_list:
        if idx in q_set or idx in f_set:
            continue
        if postags[idx].upper() not in postag_set:
            continue
        relation_trigger_list.append((word, i_val, idx))
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
    :param  relation_trigger_list 关系候选词语列表[(word, i_val)]
    :param  word2vec_model  训练好的word2vec模型
    :return 候选关系词语向量 n * 100
    '''
    word_vec = []
    for word, _, _ in relation_trigger_list:
        if word in word2vec_model:
            word_vec.append(word2vec_model[word])
        else:
            word_vec.append([0] * 100)
    return np.array(word_vec)


def get_trigger_by_ap_cluster(trigger_candidate, i_vector):
    '''
    通过AP聚类的结果得到关系trigger表示词语
    :param  trigger_candidate  关系候trigger选词语列表[(word, i_val, idx)]
    :param  i_vector  I向量
    :return 关系trigger表示词语(word, i_val, idx)
    '''
    if len(trigger_candidate) == 1:
        return trigger_candidate[0]
    trigger_i_vector = [i_vector[idx] for _, _, idx in trigger_candidate]
    ap = AffinityPropagation().fit(trigger_i_vector)
    cluster = dict()
    for (word, i_val, idx), label in zip(trigger_candidate, ap.labels_):
        c = cluster.get(label, [])
        c.append((word, i_val, idx))
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


def get_trigger_neighbour_list_from_sents(sents, entity_relation_list, save_file=None, params: Param=None):
    '''
    根据句子列表得到trigger周围词语集合，并根据需要保存成文件以免重复计算, 格式[[word1, trigger, word3], [...]]
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
    trigger_neighbour = 3
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
    if hasattr(params, 'trigger_neighbour'):
        trigger_neighbour = params.trigger_neighbour
    if hasattr(params, 'pyltp_model'):
        pyltp_model = params.pyltp_model
    if hasattr(params, 'user_dict'):
        user_dict = params.user_dict


    if save_file and os.path.exists(save_file):
        return pkl.load(open(save_file, 'rb'))
    else:
        # 得到trigger
        nlp_tool = get_nlp_tool(style, pyltp_model, user_dict)
        data_list = []
        for sent, entity_relation in zip(sents, entity_relation_list):
            word_list, postags, dependency_tree = words_postags_dependency_tree(
                sent,
                nlp_tool,
                style=style
            )
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
            trigger_neighbour_words = get_trigger_neighbour_words(trigger, word_list, trigger_neighbour)
            data_list.append(trigger_neighbour_words)
        if style == 'stanfordcorenlp':
            nlp_tool.close()
        if save_file:
            dirname = os.path.dirname(save_file)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            pkl.dump(data_list, open(save_file, 'wb'))
        return data_list


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
    feature_vector = np.zeros((1, vector_size))
    for trigger_vector in trigger_neighbour_vector_list[0]:
        feature_vector += trigger_vector
    feature_vector /= len(trigger_neighbour_vector_list[0])
    for i in range(1, len(trigger_neighbour_vector_list)):
        tv = np.zeros((1, vector_size))
        for trigger_vector in trigger_neighbour_vector_list[i]:
            tv += trigger_vector
        tv /= len(trigger_neighbour_vector_list[i])
        feature_vector = np.concatenate([feature_vector, tv], axis=0)
    return feature_vector.astype(np.float32)


def get_trigger_neighbour_words(trigger, word_list, trigger_neighbour):
    n = trigger_neighbour // 2
    len_word_list = len(word_list)
    if trigger_neighbour <= len_word_list and trigger_neighbour > 0:
        if trigger[2] - n >= 0 and trigger[2] + n < len_word_list:
            return word_list[trigger[2]-n:trigger[2]+n+1]
        elif trigger[2] - n < 0:
            return word_list[:trigger_neighbour]
        elif trigger[2] + n >= len_word_list:
            return word_list[len_word_list-trigger_neighbour:]
    return word_list


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
    to_idx = min([data_num, from_idx + batch_size])
    print(from_idx, '---------', to_idx)
    return data[from_idx:to_idx, :]


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