import tensorflow as tf
import numpy as np

def gradient_basic():
    def cost(x, y, w, b):
        c = 0
        for i in range(len(x)):
            hx = w*x[i] + b
            c += (hx - y[i]) ** 2
        return c / len(x)

    def gradient_descent(x, y, w, b):
        c1, c2 = 0, 0
        for i in range(len(x)):
            hx = w*x[i] + b
            c1 += (hx - y[i]) * x[i]
            c2 += (hx - y[i])
        return c1 / len(x), c2 / len(x)

    x = [1, 2, 3]
    y = [1, 2, 3]

    w, b = 10, -10
    for i in range(1000):
        c = cost(x, y, w, b)
        g1, g2 = gradient_descent(x, y, w, b)

        w -= 0.1 * g1
        b -= 0.1 * g2


    print('{:.2f} : {:.2f} : {:.2f}'.format(c, w, b))






def cost(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w[0]*x[i][0] + w[1]*x[i][1] + w[2]*x[i][2]
        c += (hx - y[i]) ** 2
    return c / len(x)

def gradient_descent(x, y, w):
    c1, c2, c3 = 0, 0, 0
    for i in range(len(x)):
        hx = w[0] * x[i][0] + w[1] * x[i][1] + w[2] * x[i][2]
        c1 += (hx - y[i]) * x[i][0]
        c2 += (hx - y[i]) * x[i][1]
        c3 += (hx - y[i]) * x[i][2]
    return c1 / len(x), c2 / len(x), c3 / len(x)

def show_result(x, y, w):
    for i in range(1000):
        c = cost(x, y, w)
        g1, g2, g3 = gradient_descent(x, y, w)

        w[0] -= 0.1 * g1
        w[1] -= 0.1 * g2
        w[2] -= 0.1 * g3
    print('{:.2f} : {:.2f} : {:.2f} : {:.2f}'.format(c, w[0], w[1], w[2]))
    # print('{:.2f} : {:.2f} : {:.2f}'.format(c, w, b))


x = [[1., 1., 0.], [1., 0., 2.], [1., 3., 0.], [1., 0., 4.], [1., 5., 0.]]
y = [1, 2, 3, 4, 5]

show_result(x, y, [10, -10, 10])
show_result(x, y, [10, 10, -10])

