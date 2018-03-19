import tensorflow as tf
import numpy as np
import csv
import random

def get_iris(sp_true, sp_false):
    f = open('Data/Data/iris.csv', 'r', encoding='utf-8')

    f.readline()

    iris = []
    for row in csv.reader(f):
        # print(row)

        if row[-1] == sp_true or row[-1] == sp_false:
            item = [1.]
            item += [float(i) for i in row[1: -1]]
            item.append(1 if row[-1] == sp_true else 0)
            iris.append(item)

    f.close()
    return iris

# train_set은 70개, test_set은 30개의 데이터를 작도록 분할해서 반환
def get_train_set(iris):
    # 방법 2
    # train_set = iris[:35] + iris[-35:]
    # test_set = iris[35:-35]
    #
    # return np.array(train_set), np.array(test_set)

    # 방법 3
    random.shuffle(iris)
    train_set = iris[:70]
    test_set = iris[70:]
    return np.array(train_set), np.array(test_set)

    # 방법 4
    # iris = np.array(iris)
    # # train_set = iris[:35] + iris[-35:]        -> 이거는 vector 연산, 진짜 각 원소간 덧셈으로 처리됌
    # train_set = np.vstack([iris[:35], iris[-35:]])
    # test_set = iris[35:-35]
    #
    # return train_set, test_set

    # 방법 5
    # iris = np.array(iris)
    #
    # np.random.shuffle(iris)
    #
    # train_set = iris[:70]
    # test_set = iris[70:]
    #
    # return train_set, test_set

def count(x, y):
    cnt = 0
    for i in range(30):
        if x[i] == y[i]:
            cnt += 1

    print((cnt/30)*100, '%')



iris = get_iris('setosa', 'versicolor')

# print(*iris, sep='\n')

train_set, test_set = get_train_set(iris)

print(train_set.shape, test_set.shape)

xx = train_set[:, :-1]
y = train_set[:, -1:]

xx_test = test_set[:, :-1]
y_test = test_set[:, -1]

print(xx.shape, y.shape)

x = tf.placeholder(tf.float32, shape=[None, 5])
w = tf.Variable(tf.random_uniform([5, 1], -1, 1))

z = tf.matmul(x, w)
hx = tf.nn.sigmoid(z)
cost = tf.reduce_mean(   y  * -tf.log(  hx) +
                      (1-y) * -tf.log(1-hx))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss=cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10):
    sess.run(train, {x: xx})
    print(i, sess.run(cost, {x: xx}))

y_hat = (sess.run(hx, {x: xx_test}))
res = (y_hat >= 0.5)
res = res.reshape(-1)

count(res, y_test.reshape(-1))

sess.close()