import tensorflow as tf
import numpy as np
from sklearn import model_selection

def get_iris():
    iris = np.loadtxt('Data/Data/iris_softmax.csv', delimiter=',')

    x_data = iris[:, :-3]
    y_data = iris[:, -3:]

    data = model_selection.train_test_split(x_data, y_data, train_size=120, test_size=30)
    x_train, x_test, y_train, y_test = data

    return x_train, x_test, y_train, y_test

class Simple:
    def __init__(self, x_train, x_test, y_train, y_test):

        x = tf.placeholder(tf.float32, shape=[None, 5])
        w = tf.Variable(tf.random_normal([5, 3], -1, 1))

        z = tf.matmul(x, w)

        hypothesis = tf.nn.softmax(z)

        cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y_train)

        cost = tf.reduce_mean(cost_i)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train = optimizer.minimize(loss=cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(100):
            sess.run(train, {x: x_train})
            # print(i, sess.run(cost, {x: x_train}))

        self.y_hat = sess.run(hypothesis, {x: x_test})
        self.y_test = y_test

        sess.close()

    def show_accuracy(self):
        y_hat_bool = np.argmax(self.y_hat, axis=1)
        y_test_bool = np.argmax(self.y_test, axis=1)

        equals = (y_hat_bool == y_test_bool)
        print('accracy = ', np.mean(equals))


class SimpleModel:
    def __init__(self, count, x_train, x_test, y_train, y_test):
        self.models = [Simple(x_train, x_test, y_train, y_test) for i in range(count)]

        self.y_test = y_test

    def show_accuracy(self):
        result = np.zeros_like(self.y_test)
        for model in self.models:
            model.show_accuracy()
            result += model.y_hat

        print('-'*50)

        y_hat_bool = np.argmax(result, axis=1)
        y_test_bool = np.argmax(self.y_test, axis=1)

        equals = (y_hat_bool == y_test_bool)
        print('accracy = ', np.mean(equals))








x_train, x_test, y_train, y_test = get_iris()

# simple = Simple(x_train, x_test, y_train, y_test)
#
# simple.show_accuracy()
# # Simple.show_accuracy(simple)  --> 같은방법임

model = SimpleModel(7, x_train, x_test, y_train, y_test)
model.show_accuracy()



