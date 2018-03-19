import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def loadtxt():
    cars = np.loadtxt('Data/Data/cars.csv', delimiter=',', dtype=np.float32)
    print(cars)
    print(type(cars))  # <class 'numpy.ndarray'>
    print(cars.shape)  # (50, 2)
    print(cars[0])
    print(cars[0][0])
    print(cars.dtype)  # default = float64

# 문제
# 리니어 리그레션 기본 코드를 사용해서
# cars.csv 파일에 적용해보시오
def get_cars():
    cars = np.loadtxt('Data/Data/cars.csv', delimiter=',', dtype=np.float32)

    # #version 1
    # x, y = [], []
    # for speed, dist in cars:
    #     x.append(speed)
    #     y.append(dist)
    # return x, y

    # version 2
    # return cars.T 라고 쓸 수 도 있다. 아래와 똑같은 코드임
    # cars.transpose()
    # return cars

    # version 3
    return cars[:, 0], cars[:, 1]


    # version 4
    # cars = np.loadtxt('Data/Data/cars.csv', delimiter=',', dtype=np.float32, unpack=True)
    # print(cars.shape)
    # return cars


def regression_1():
    x, y = get_cars()

    w = tf.Variable(tf.random_normal([1], -1, 1))
    b = tf.Variable(tf.random_normal([1], -1, 1))

    hypothesis = w * x + b
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss=cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()

# 문제
# 속도가 30과 50일 때의 제동거리를 예측해보세요
# placeholder 사용

def regression_2():
    xx, y = get_cars()

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_normal([1], -1, 1))
    b = tf.Variable(tf.random_normal([1], -1, 1))

    hypothesis = w * x + b
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss=cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {x: xx})
        print(i, sess.run(cost, {x: xx}))

    new_xx = [30, 50]
    print(sess.run(hypothesis, {x: new_xx}))

    # plt.plot(xx, y)
    plt.plot(xx, y, 'ro')
    plt.plot([0, 30], [sess.run(hypothesis, {x: 0}), sess.run(hypothesis, {x: 30})])
    plt.show()

    sess.close()

regression_2()