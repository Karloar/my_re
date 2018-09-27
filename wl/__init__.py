# 将Pyltp生成的arcs转换成依存关系树
from wl.myfuncs import arcs_to_dependency_tree

# 通过PageRank计算 I(v|R)
from wl.myfuncs import page_rank

# 计算 I(v|{Q, F})
from wl.myfuncs import get_I_vector

# 得到根据I_val降序排列的词列表 (词语, I值, 索引)
from wl.myfuncs import get_sorted_word_I_list

# 从分词后的词列表中找到人物实体
from wl.myfuncs import get_person_entity_set

# 返回代表关系的候选词语
from wl.myfuncs import get_trigger_candidate

# 得到实体1和实体2的修饰词集合
from wl.myfuncs import get_modifier_set

from wl.myfuncs import dependency_tree_to_arr

# 初始化Pyltp的几个模块 segmentor, postagger, parser, ner
from wl.myfuncs import init_pyltp

# 得到候选关系词语的词向量
from wl.myfuncs import get_trigger_candidate_vector

# 通过AP聚类的结果得到关系表示词语 (word, i_val)
from wl.myfuncs import get_trigger_by_ap_cluster

# 从工程根目录开始找到文件路径
from wl.myfuncs import get_resource_path
