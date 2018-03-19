import math
import matplotlib.pyplot as plt
import numpy as np

def step_function(x):
    y = (x > 0)
    return np.int32(y)

def activation_1():
    x = np.arange(-5, 5, 0.1)
    y = step_function(x)

    plt.plot(x, y)
    plt.show()


def activation_2():
    def sigmoid(z):
        return 1 / (1 + math.e ** -z)

    print(sigmoid(-5))
    print(sigmoid(-1))
    print(sigmoid(0))
    print(sigmoid(1))
    print(sigmoid(5))

    for z in range(-10, 10):
        plt.plot(z, sigmoid(z), 'ro')
    plt.show()

def A():
    return 'A'

def B():
    return 'B'

def binding():
    y = 1
    if y == 1:
        print(A())
    else:
        print(B())

    # 아래 코드가 성립할 수 있는 전제조건 : y가 0 or 1 둘중 하나의 값을 가진다
    print(y*A() + (1-y)*B())

    y = 0
    if y == 1:
        print(A())
    else:
        print(B())

    print(y*A() + (1-y)*B())

