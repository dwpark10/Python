import tensorflow as tf
import numpy as np

def logistic_regression():
    x = [[1., 1., 1., 1., 1., 1.],
         [2., 3., 3., 5., 7., 2.],
         [1., 2., 5., 5., 5., 5.]]
    y = np.array([0, 0, 0, 1, 1, 1])

    w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

    z = tf.matmul(w, x)
    hx = 1 / (1 + tf.exp(-z))
    cost = tf.reduce_mean(   y  * -tf.log(  hx) +
                          (1-y) * -tf.log(1-hx))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

# 예측
xx = [[1., 1., 1., 1., 1., 1.],
     [2., 3., 3., 5., 7., 2.],
     [1., 2., 5., 5., 5., 5.]]
y = np.array([0, 0, 0, 1, 1, 1])

x = tf.placeholder(tf.float32, shape=[3, None])
w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

z = tf.matmul(w, x)
# hx = 1 / (1 + tf.exp(-z))
hx = tf.nn.sigmoid(z)
cost = tf.reduce_mean(   y  * -tf.log(  hx) +
                      (1-y) * -tf.log(1-hx))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss=cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train, {x: xx})
    print(i, sess.run(cost, {x: xx}))

y_hat = sess.run(hx, {x: [[1., 1.], [7., 3.], [2., 6.]]})
print(y_hat)
print(y_hat >= 0.5)

sess.close()