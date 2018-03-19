import tensorflow as tf
import numpy as np
from sklearn import model_selection


def get_iris():
    iris = np.loadtxt('Data/iris_softmax.csv', delimiter=',')

    x_data = iris[:, :-3]
    y_data = iris[:, -3:]

    data = model_selection.train_test_split(x_data, y_data,
                                            train_size=120,
                                            test_size=30)
    x_train, x_test, y_train, y_test = data
    return x_train, x_test, y_train, y_test


class Better:
    def __init__(self, sess, x_train, x_test, y_train, y_test,
                 learning_rate=0.1, loop_count=100, show_cost=False):
        self.sess = sess
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.loop_count = loop_count
        self.show_cost = show_cost

        # --------------------------- #

        n_features = x_train.shape[-1]
        n_classes = y_train.shape[-1]

        self.x = tf.placeholder(tf.float32)
        w = tf.Variable(tf.random_normal([n_features, n_classes], -1, 1))

        z = tf.matmul(self.x, w)
        self.hypothesis = tf.nn.softmax(z)

        cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z,
                                                         labels=y_train)
        self.cost = tf.reduce_mean(cost_i)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train = optimizer.minimize(loss=self.cost)

    def initialize(self):
        # sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def run(self):
        for i in range(self.loop_count):
            self.sess.run(self.train, {self.x: self.x_train})

            if self.show_cost:
                print(i, self.sess.run(self.cost, {self.x: self.x_train}))

    def predict(self):
        self.y_hat = self.sess.run(self.hypothesis, {self.x: self.x_test})
        # self.y_test = y_test

        # sess.close()

    def show_accuracy(self):
        y_hat_bool = np.argmax(self.y_hat, axis=1)
        y_test_bool = np.argmax(self.y_test, axis=1)

        equals = (y_hat_bool == y_test_bool)
        print('accuracy :', np.mean(equals))
        # print(self.y_hat[0])


class BetterModel:
    def __init__(self, sess, count, x_train, x_test, y_train, y_test,
                 learning_rate=0.1, loop_count=100, show_cost=False):
        self.models = [Better(sess, x_train, x_test, y_train, y_test,
                              learning_rate, loop_count, show_cost)
                       for _ in range(count)]
        self.y_test = y_test
        self.sess = sess

        sess.run(tf.global_variables_initializer())

    def show_accuracy(self):
        result = np.zeros_like(self.y_test)
        for model in self.models:
            # model.initialize()
            model.run()
            model.predict()
            model.show_accuracy()
            result += model.y_hat

        print('-' * 50)

        y_hat_bool = np.argmax(result, axis=1)
        y_test_bool = np.argmax(self.y_test, axis=1)

        equals = (y_hat_bool == y_test_bool)
        print('accuracy :', np.mean(equals))


x_train, x_test, y_train, y_test = get_iris()

with tf.Session() as sess:
    # better = Better(sess, x_train, x_test, y_train, y_test)
    # better.initialize()
    # better.run()
    # better.predict()
    # better.show_accuracy()

    model = BetterModel(sess, 7, x_train, x_test, y_train, y_test)
    model.show_accuracy()