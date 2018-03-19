import numpy as np
import matplotlib.pyplot as plt
import random

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(x, y):
    m, n = x.shape
    w = np.zeros([n, 1])
    lr = 0.01

    for _ in range(m):
        z = np.dot(x, w)        # (100, 1) = (100, 3) x (3, 1)
        h = sigmoid(z)          # (100, 1)
        e = h - y               # (100, 1) = (100, 1) - (100, 1)
        g = np.dot(x.T, e)     # (3, 1)   = (3, 100) x (100, 1)
        w -= lr * g             # (3, 1)  -= (3, 1)

    return w.reshape(-1)        # (3, )

def stochastic(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros(n)             # (3, )
    lr = 0.01

    for i in range(m * 10):
        p = i % m
        z = np.sum(x[p] * w)    # scalar = sum((3, ) * (3, ))
        h = sigmoid(z)          # scalar
        e = h - y[p]            # scalar = scalar - scalar
        g = x[p] * e            # (3, )   = (3, ) * scalar
        w -= lr * g             # (3, )  -= scalar * (3, )

    return w                    # (3, )

def stochastic_random(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros(n)             # (3, )
    lr = 0.01

    for i in range(m * 10):
        p = random.randrange(m)
        z = np.sum(x[p] * w)    # scalar = sum((3, ) * (3, ))
        h = sigmoid(z)          # scalar
        e = h - y[p]            # scalar = scalar - scalar
        g = x[p] * e            # (3, )   = (3, ) * scalar
        w -= lr * g             # (3, )  -= scalar * (3, )

    return w                    # (3, )

def mini_batch(x, y):
    m, n = x.shape
    w = np.zeros([n, 1])
    lr = 0.01
    epochs = 10
    batch_size = 5

    count = m // batch_size
    for _ in range(epochs):

        for k in range(count):
            s = k * batch_size
            f = s + batch_size

            z = np.dot(x[s:f], w)   # (5, 1) = (5, 3) x (3, 1)
            h = sigmoid(z)          # (5, 1)
            e = h - y[s:f]          # (5, 1) = (5, 1) - (5, 1)
            g = np.dot(x[s:f].T, e) # (3, 1)   = (3, 5) x (5, 1)
            w -= lr * g             # (3, 1)  -= (3, 1)

    return w.reshape(-1)        # (3, )

def mini_batch_random(x, y):        # 셔플하기 편하게 하기 위해서 원래는 x, y를 하나로 합쳐서 전달한다
    m, n = x.shape
    w = np.zeros([n, 1])
    lr = 0.01
    epochs = 10
    batch_size = 5

    count = m // batch_size
    for _ in range(epochs):

        for k in range(count):
            s = k * batch_size
            f = s + batch_size

            z = np.dot(x[s:f], w)   # (5, 1) = (5, 3) x (3, 1)
            h = sigmoid(z)          # (5, 1)
            e = h - y[s:f]          # (5, 1) = (5, 1) - (5, 1)
            g = np.dot(x[s:f].T, e) # (3, 1)   = (3, 5) x (5, 1)
            w -= lr * g             # (3, 1)  -= (3, 1)

        seed = random.randrange(1000)
        np.random.seed(seed)
        np.random.shuffle(x)

        np.random.seed(seed)
        np.random.shuffle(y)

    return w.reshape(-1)        # (3, )

def decision_boundary(w, c):
    b, w1, w2 = w[0], w[1], w[2]

    # hx = w1 * x1 + w2 * x2 + b
    # 0 = w1 * x1 + w2 * x2 + b     --> sigmoid 를 쓰고 있으니까.. 0일때 데이터들의 중간을 가로지르는 선이 그려진다
    # 0 = w1 * x + w2 * y + b       --> 좌표축 상에서 x, y
    # -(w1 * x + b) = w2 * y
    # y = -(w1 * x + b) / w2

    y1 = -(w1 * -4 + b) / w2
    y2 = -(w1 *  4 + b) / w2

    plt.plot([-4, 4], [y1, y2], c)



action = np.loadtxt('Data/Data/action.txt', delimiter=',')
print(action.shape)

xx = action[:, :-1]
yy = action[:, -1:]
print(xx.shape, yy.shape)

for _, x1, x2, y in action:
    plt.plot(x1, x2, 'ro' if y else 'go')

decision_boundary(gradient_descent(xx, yy), 'r')
decision_boundary(stochastic(xx, yy), 'g')
decision_boundary(stochastic_random(xx, yy), 'b')
decision_boundary(mini_batch(xx, yy), 'k')
decision_boundary(mini_batch_random(xx, yy), 'y')
plt.show()