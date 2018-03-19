import tensorflow as tf
import numpy as np
from sklearn import preprocessing

trees = np.loadtxt('Data/Data/trees.csv', delimiter=',', dtype=np.float32, skiprows=1)

# trees = preprocessing.add_dummy_feature(trees)
trees = np.insert(trees, 0, np.ones(31), axis=1)

xx = trees[:, :-1]
y = trees[:, -1:]
print(xx.shape, y.shape)

x = tf.placeholder(tf.float32, shape=[None, 3])
w = tf.Variable(tf.random_normal([3, 1], -1, 1))

hx = tf.matmul(x, w)
cost = tf.reduce_mean((hx - y) ** 2)

optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10):
    sess.run(train, {x: xx})
    print(i, sess.run(cost, {x: xx}))

y_hat = sess.run(hx, {x: xx})
print(y_hat)
y_hat = sess.run(hx, {x: [[1., 10., 70.], [1., 15., 80.]]})
print(y_hat)

sess.close()