import tensorflow as tf
import numpy as np

def normalize(x):
    # (나 - 최소값) / (최대값 - 최소값)      // min-max scaling
    mn = np.min(x, axis=0)
    mx = np.max(x, axis=0)
    print(mn)
    print(mx)

    return (x - mn) / (mx - mn)

def test_normalize(use):
    data = [[828, 833, 1908100, 828, 831],
            [823, 827, 1828100, 821, 826],
            [819, 824, 1438100, 818, 823],
            [816, 820, 1008100, 815, 819],
            [819, 823, 1188100, 817, 833],
            [840, 850, 1234100, 822, 818],
            [811, 814, 1098100, 809, 813],
            [850, 860, 1534100, 830, 840]]

    data = np.float32(data)

    if use:
        data = normalize(data)

    print(data)

    x = data[:, :-1].T
    y = data[:, -1]

    print(x.shape, y.shape)

    w = tf.Variable(tf.random_uniform([1, 4], -1, 1))

    hx = tf.matmul(w, x)
    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))


test_normalize(True)