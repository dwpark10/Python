import tensorflow as tf
import numpy as np

def multy_regression_1():
    x1 = [1, 0, 3, 0, 5]
    x2 = [0, 2, 0, 4, 0]
    y = [1, 2, 3, 4, 5]

    w1 = tf.Variable(tf.random_normal([1], -1, 1))
    w2 = tf.Variable(tf.random_normal([1], -1, 1))
    b = tf.Variable(tf.random_normal([1], -1, 1))

    hx = w1 * x1 + w2 * x2 + b
    cost = tf.reduce_mean((hx-y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()

def multy_regression_2():
    # 문제
    # x를 변수 하나로 만든 다음에 코드가 동작하도록
    # 행렬 곱셈 : tf.matmul()
    x = [[1., 0., 3., 0., 5.], [0., 2., 0., 4., 0.]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_normal([1, 2], -1, 1))
    b = tf.Variable(tf.random_normal([1], -1, 1))

    # hx = w[0] * x[0] + w[1] * x[1] + b
    # (1, 5) = (1, 2) * (2, 5)
    hx = tf.matmul(w, x) + b
    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()

def multy_regression_3():
# 문제
# bias 를 없애보세요
    x = [[1., 0., 3., 0., 5.], [0., 2., 0., 4., 0.], [1., 1., 1., 1., 1.]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_normal([1, 3], -1, 1))

    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # (1, 5) = (1, 2) * (2, 5)
    hx = tf.matmul(w, x)
    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    print(sess.run(w))
    sess.close()

def multy_regression_4():
    # bias 의 값은 사실상 어디 들어가든 상관이 없다.
    # 일반적으로 맨 앞에 놓는다
    # 뒤에 얼마나 많은 weight 가 들어올지 모르니까
    x = [[1., 1., 1., 1., 1.], [1., 0., 3., 0., 5.], [0., 2., 0., 4., 0.]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_normal([1, 3], -1, 1))

    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # (1, 5) = (1, 2) * (2, 5)
    hx = tf.matmul(w, x)
    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    print(sess.run(w))
    sess.close()


def multy_regression_6():
    # 문제
    # 행렬 곱셈에서 w와 x의 위치를 바꾸세요
    # x = [[1., 1., 1., 1., 1.], [1., 0., 3., 0., 5.], [0., 2., 0., 4., 0.]]
    x = [[1., 1., 0.], [1., 0., 2.], [1., 3., 0.], [1., 0., 4.], [1., 5., 0.]]
    y = [[1.], [2.], [3.], [4.], [5.]]


    w = tf.Variable(tf.random_normal([3, 1], -1, 1))

    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # (1, 5) = (1, 3) * (3, 5)

    # hx = x[0] * w[0] + x[1] * w[1] + x[2] * w[2]
    # (1, 5) = (3, 5) * (3, 1)

    hx = tf.matmul(x, w)
    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    print(sess.run(w))
    sess.close()


# 문제
# 5시간 공부하고 7번 출석한 경우와
# 7시간 공부하고 4번 출석한 경우에 대해 예측해 보시오
xx = [[1., 1., 0.], [1., 0., 2.], [1., 3., 0.], [1., 0., 4.], [1., 5., 0.]]
y = [[1.], [2.], [3.], [4.], [5.]]

# 보통 numpy 에서는 지정하지 않는 값에 대해서 -1 을 쓰지만
# tensorflow 에서는 None 으로 준다
x = tf.placeholder(tf.float32, shape=[None, 3])


w = tf.Variable(tf.random_normal([3, 1], -1, 1))

# hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
# (1, 5) = (1, 3) * (3, 5)

# hx = x[0] * w[0] + x[1] * w[1] + x[2] * w[2]
# (1, 5) = (3, 5) * (3, 1)

hx = tf.matmul(x, w)
cost = tf.reduce_mean((hx - y) ** 2)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10):
    sess.run(train, {x: xx})
    print(i, sess.run(cost, {x: xx}))

y_hat = sess.run(hx, {x: xx})
print(y_hat)
y_hat = sess.run(hx, {x: [[1., 5., 7.], [1., 7., 4.]]})
print(y_hat)

sess.close()


