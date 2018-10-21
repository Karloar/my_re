import tensorflow as tf
from myfuncs import get_trigger_neighbour_list_from_sents
from myfuncs import get_resource_path
from myfuncs import load_data_en
from myfuncs import print_running_time
from myfuncs import get_trigger_neighbour_vector_list
from myfuncs import get_feature_vector_for_nn
from myfuncs import Param
import os
import platform
from gensim.models import Word2Vec
import numpy as np
from sklearn.neural_network import MLPClassifier


model_dir = '/Users/karloar/Documents/other/ltp_data_v3.4.0'
if platform.system() == 'Windows':
    model_dir = r'E:\ltp_data'
train_file = get_resource_path("data/TRAIN_FILE.TXT")
test_file = get_resource_path("data/TEST_FILE_FULL.TXT")
word2vec_model_file = os.path.join(model_dir, 'wiki_en_vector_vec.model')

vector_size = 200
train_neighbour_words_pkl = 'saved/train_neighbour_words.pkl'
test_neighbour_words_pkl = 'saved/test_neighbour_words.pkl'



class MyNN:

    def __init__(
        self,
        batch_size,
        layer,
        activation,
        learning_rate=0.01,
        optimizer=tf.train.GradientDescentOptimizer,
        step_num=3000
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer(learning_rate)
        self.layer = layer
        self.activation = activation
        self.step_num = step_num

    def train(self, X, y):
        with tf.name_scope("input_data"):
            self.xs = tf.placeholder(tf.float32, (None, X.shape[1]), "input_data")
            self.ys = tf.placeholder(tf.float32, (None, y.shape[1]), "y")
        # hidden_layer_input = self.add_layer(X, X.shape[1], self.layer[0], self.activation[0])
        layer = [X.shape[1]]
        for s in self.layer:
            layer.append(s)
        layer.append(y.shape[1])
        hidden_layer_input = X
        for i in range(len(layer) - 1):
            hidden_layer_input = self.add_layer(hidden_layer_input, layer[i], layer[i+1], self.activation[i])
        # for i in range(len(self.activation[1:-1])):
        #     layer_dim, activation
        #     hidden_layer_input = self.add_layer(
        #         hidden_layer_input,
        #         hidden_layer_input.shape[1].value,
        #         layer_dim,
        #         activation
        #     )
        # self.prediction = self.add_layer(
        #     hidden_layer_input,
        #     hidden_layer_input.shape[1].value,
        #     y.shape[1],
        #     self.activation[-1]
        # )
        self.prediction = hidden_layer_input
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction)))
        train_step = self.optimizer.minimize(cross_entropy)
        with tf.device("/gpu:0"):
            self.sess = tf.Session()
            tf.global_variables_initializer().run(session=self.sess)
            for step in range(self.step_num):
                batch_xs, batch_ys = self.get_batch(X, y, step)
                self.sess.run(train_step, feed_dict={self.xs: batch_xs, self.ys: batch_ys})
                if step % 100 == 0:
                    print(step, ":", self.sess.run(cross_entropy, feed_dict={self.xs: batch_xs, self.ys: batch_ys}))
    
    def predict(self, X, y):
        with tf.device("/gpu:0"):
            return self.compute_accuracy(X, y, self.sess, self.prediction)

    def close(self):
        self.sess.close()
    
    def get_batch(self, X, y, step):
        total = len(X)
        batch_num = int(total / self.batch_size + 0.5)
        begin_idx = step % batch_num
        end_idx = min([begin_idx + self.batch_size, total])
        return X[begin_idx:end_idx, :], y[begin_idx:end_idx, :]

    def add_layer(self, input_data, in_size, out_size, activation=None):
        with tf.name_scope("weight"):
            weights = tf.Variable(tf.random_uniform((in_size, out_size)), dtype=tf.float32, name="weight")
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.ones((1, out_size)), dtype=tf.float32, name="biase")
        wx_plus_b = tf.matmul(input_data, weights) + biases
        if activation:
            return activation(wx_plus_b)
        return wx_plus_b
    
    def compute_accuracy(self, x, y, session, prediction):
        y_pre = session.run(prediction, feed_dict={self.xs: x, self.ys: y})
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pre, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = session.run(accuracy, feed_dict={self.xs: x, self.ys: y})
        return result
    

def get_label_by_entity_relation_list(entity_relation_list: list, relation_list: list):
    m = len(entity_relation_list)
    n = len(relation_list)
    label = np.zeros((m, n)).astype(np.float32)
    for i in range(m):
        idx = relation_list.index(entity_relation_list[i][1])
        label[i, idx] = 1
    return label


@print_running_time
def main():
# 加载word2vec模型
    model = Word2Vec.load(word2vec_model_file)

    # 设置参数
    params = Param()
    params.trigger_neighbour = 3

    # 处理训练数据
    print('processing train data......')
    sents, entity_relation_list = load_data_en(train_file)
    relation_list = list(set(x[1] for x in entity_relation_list))
    trigger_neighbour_list = get_trigger_neighbour_list_from_sents(
        sents, entity_relation_list,
        os.path.join(os.getcwd(), train_neighbour_words_pkl),
        params=params
    )
    trigger_neighbour_vector_list = get_trigger_neighbour_vector_list(
        trigger_neighbour_list, model, vector_size
    )
    train_data = get_feature_vector_for_nn(
        trigger_neighbour_vector_list, vector_size
    )
    train_label = get_label_by_entity_relation_list(entity_relation_list, relation_list)

    # 处理训练数据
    print('processing test data......')
    sents, entity_relation_list = load_data_en(test_file)
    trigger_neighbour_list = get_trigger_neighbour_list_from_sents(
        sents, entity_relation_list,
        os.path.join(os.getcwd(), test_neighbour_words_pkl),
        params=params
    )
    trigger_neighbour_vector_list = get_trigger_neighbour_vector_list(
        trigger_neighbour_list, model, vector_size
    )
    test_data = get_feature_vector_for_nn(
        trigger_neighbour_vector_list, vector_size
    )
    test_label = get_label_by_entity_relation_list(entity_relation_list, relation_list)

    mynn = MyNN(
        batch_size=100,
        layer=(400,),
        activation=(tf.nn.relu, tf.nn.softmax)
    )
    mynn.train(train_data, train_label)
    print(mynn.predict(test_data, test_label))


if __name__ == '__main__':
    main()
