import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# python 문법
# class 안에서 @property 붙이는거 -> 함수이지만 사용할때는 ()를 붙이지 않고 변수를 쓰는것처럼 사용한다
# 한번 찾아보기

def show_accuray(hx, sess, x, y, keep_rate, prompt, dataset):
    pred = tf.equal(tf.argmax(hx, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

    print(prompt, sess.run(accuracy, {x: dataset.images, y: dataset.labels, keep_rate: 1.0}))


def softmax(x, y, _):
    w = tf.Variable(tf.zeros([784, 10]))  # 784 = 28 * 28 / 이미지 사이즈
    b = tf.Variable(tf.zeros([10]))

    z = tf.matmul(x, w) + b
    hx = tf.nn.softmax(z)

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)

    cost = tf.reduce_mean(cost_i)

    return hx, cost


# hyper parameter : 1. layer의 갯수
#                   2. neuron 의 갯수
def multi_layer_relu(x, y, _):
    w1 = tf.Variable(tf.random_normal([784, 256]))  # 784 = 28 * 28 / 이미지 사이즈
    w2 = tf.Variable(tf.random_normal([256, 256]))
    w3 = tf.Variable(tf.random_normal([256, 10]))

    b1 = tf.Variable(tf.zeros([256]))
    b2 = tf.Variable(tf.zeros([256]))
    b3 = tf.Variable(tf.zeros([10]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)

    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)

    hx = tf.matmul(r2, w3) + b3

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=hx, labels=y)

    cost = tf.reduce_mean(cost_i)

    return hx, cost

def multi_layer_xavier_1(x, y, _):
    w1 = tf.get_variable('w1', shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable('w2', shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
    w3 = tf.get_variable('w3', shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())

    b1 = tf.Variable(tf.zeros([256]))
    b2 = tf.Variable(tf.zeros([256]))
    b3 = tf.Variable(tf.zeros([10]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)

    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)

    hx = tf.matmul(r2, w3) + b3

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=hx, labels=y)

    cost = tf.reduce_mean(cost_i)

    return hx, cost

def multi_layer_xavier_2(x, y, _):
    w1 = tf.get_variable('w1_', shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable('w2_', shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
    w3 = tf.get_variable('w3_', shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())

    b1 = tf.Variable(tf.random_normal([256]))
    b2 = tf.Variable(tf.random_normal([256]))
    b3 = tf.Variable(tf.random_normal([10]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)

    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)

    hx = tf.matmul(r2, w3) + b3

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=hx, labels=y)

    cost = tf.reduce_mean(cost_i)

    return hx, cost

def multi_layer_dropout(x, y, keep_rate):
    w1 = tf.get_variable('w1__', shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable('w2__', shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
    w3 = tf.get_variable('w3__', shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
    w4 = tf.get_variable('w4__', shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
    w5 = tf.get_variable('w5__', shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())


    b1 = tf.Variable(tf.zeros([256]))
    b2 = tf.Variable(tf.zeros([256]))
    b3 = tf.Variable(tf.zeros([256]))
    b4 = tf.Variable(tf.zeros([256]))
    b5 = tf.Variable(tf.zeros([10]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)
    d1 = tf.nn.dropout(r1, keep_rate)

    z2 = tf.matmul(d1, w2) + b2
    r2 = tf.nn.relu(z2)
    d2 = tf.nn.dropout(r2, keep_rate)

    z3 = tf.matmul(d2, w3) + b3
    r3 = tf.nn.relu(z3)
    d3 = tf.nn.dropout(r3, keep_rate)

    z4 = tf.matmul(d3, w4) + b4
    r4 = tf.nn.relu(z4)
    d4 = tf.nn.dropout(r4, keep_rate)

    hx = tf.matmul(d4, w5) + b5

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=hx, labels=y)

    cost = tf.reduce_mean(cost_i)

    return hx, cost

def multi_layer_auto(x, y, keep_rate):
    layers = [784, 256, 256, 256, 256, 10]
    last = len(layers) - 1

    for i in range(last):
        w = tf.get_variable('ww_' + str(i), shape=[layers[i], layers[i+1]], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros([layers[i+1]]))

        z = tf.matmul(x, w) + b
        if i == last - 1:
            break

        r = tf.nn.relu(z)
        x = tf.nn.dropout(r, keep_rate)

    hx = z

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=hx, labels=y)

    cost = tf.reduce_mean(cost_i)

    return hx, cost



def show_model(model):
    mnist = input_data.read_data_sets('mnist', one_hot=True)

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    keep_rate = tf.placeholder(tf.float32)

    ##############################################################################################

    hx, cost = model(x, y, keep_rate)

    ##############################################################################################

    # optimizer = tf.train.GradientDescentOptimizer(0.01)
    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs, batch_size = 15, 100

    count = mnist.train.num_examples // batch_size
    for i in range(epochs):
        total_cost = 0
        for _ in range(count):
            xx, yy = mnist.train.next_batch(batch_size)

            _, c = sess.run([train, cost], {x: xx, y: yy, keep_rate: 0.7})

            total_cost += c

        print(i, total_cost/count)
    print()

    ##############################################################################################

    show_accuray(hx, sess, x, y, keep_rate, 'train :', mnist.train)
    show_accuray(hx, sess, x, y, keep_rate, 'train :', mnist.validation)
    show_accuray(hx, sess, x, y, keep_rate, 'train :', mnist.test)

    sess.close()

### gradient descent 사용 ###

# show_model(softmax)
# train : 0.9016727
# train : 0.909
# train : 0.9084

# show_model(multi_layer_relu)
# train : 0.9881818
# train : 0.9356
# train : 0.9338

# show_model(multi_layer_xavier_1)
# train : 0.9523636
# train : 0.9532
# train : 0.9508

# show_model(multi_layer_xavier_2)
# train : 0.9434
# train : 0.9446
# train : 0.9427

# show_model(multi_layer_dropout)
# train : 0.9619273
# train : 0.963
# train : 0.9586

### adam optimizer 사용 ###

# show_model(multi_layer_dropout)
# train : 0.9952
# train : 0.9804
# train : 0.9809

show_model(multi_layer_auto)