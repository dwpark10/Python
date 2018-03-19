import tensorflow as tf
import numpy as np
import math

def base_test_1():
    a = 2.0
    b = 1.0
    c = 0.1

    base1 = a + b + c
    base2 = math.e**a + math.e**b + math.e**c

    print(a / base1)
    print(b / base1)
    print(c / base1)


    print(math.e**a / base2)
    print(math.e**b / base2)
    print(math.e**c / base2)


def cross_entroy_1():
    xxy = np.loadtxt('Data/Data/softmax.txt', dtype=np.float32)

    print(xxy)

    x = xxy[:, :3]
    y = xxy[:, 3:]
    print(x.shape, y.shape)

    w = tf.Variable(tf.zeros([3, 3]))

    z = tf.matmul(x, w)
    hypothesis = tf.nn.softmax(z)

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)

    cost = tf.reduce_mean(cost_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()

# 문제
# 3시간 공부하고 7번 출석할 때
# 7시간 공부하고 3번 출석할 때
def cross_entroy_2():
    xxy = np.loadtxt('Data/Data/softmax.txt', dtype=np.float32)

    print(xxy)

    xx = xxy[:, :3]
    y = xxy[:, 3:]

    x = tf.placeholder(tf.float32, shape=[None, 3])
    w = tf.Variable(tf.zeros([3, 3]))

    z = tf.matmul(x, w)
    hypothesis = tf.nn.softmax(z)

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)

    cost = tf.reduce_mean(cost_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {x: xx})
        print(i, sess.run(cost, {x: xx}))

    test_xx = [[1., 3., 7],
               [1., 7., 3]]


    pred = sess.run(hypothesis, {x: test_xx})


    print(np.argmax(pred, axis=1))
    index = np.argmax(pred, axis=1)
    grades = np.array(['A', 'B', 'C'])

    print(grades[index])

    sess.close()

# 문제
# 행렬 곱셈에서 x와 w의 위치를 바꾸세요

xxy = np.loadtxt('Data/Data/softmax.txt', dtype=np.float32, unpack=True)

print(xxy.shape)
print(xxy)

xx = xxy[:3]
y = xxy[3:]
print(xx.shape, y.shape)

x = tf.placeholder(tf.float32, shape=[3, None])
w = tf.Variable(tf.zeros([3, 3]))

# (3, 8) = (3, 3) x (3, 8)
z = tf.matmul(w, x)

# (3, 8) - softmax(3, 8)
hx = tf.nn.softmax(z, dim=0)

# (3, 8) = (3, 8) * (3, 8)
cross_entropy = -tf.log(hx) * y

cost_i = tf.reduce_sum(cross_entropy, axis=0)

cost = tf.reduce_mean(cost_i)

with_logits = tf.nn.softmax_cross_entropy_with_logits(logits=hx, labels=y, dim=0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

np.set_printoptions(linewidth=1000)

zz = sess.run(z, {x: xx})
print(zz.shape)
print(zz)

hh = sess.run(hx, {x: xx})
print(hh.shape)
print(hh)

cc = sess.run(cross_entropy, {x: xx})
print(cc.shape)
print(cc)

ii = sess.run(cost_i, {x: xx})
print(ii.shape)
print(ii)

ww = sess.run(with_logits, {x: xx})
print(ww.shape)
print(ww)


ee = sess.run(cost, {x: xx})
print(ee.shape)
print(ee)


sess.close()