# lib 설치 방법 :
# File -> Settings -> project interpreter 들어가서 우측 + 누르고 검색, 설치

import matplotlib.pyplot as plt

def cost(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w*x[i]
        c += (hx - y[i]) ** 2
    return c / len(x)

def gradient_descent(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w*x[i]
        c += (hx - y[i]) * x[i]
    return c / len(x)

def show_cost():
    x = [1, 2, 3]
    y = [1, 2, 3]

    print(cost(x, y, 0))
    print(cost(s, y, 1))
    print(cost(x, y, 2))

    for i in range(-30, 50):
        w = i / 10
        c = cost(x, y, w)

        print(w, c)
        plt.plot(w, c, 'ro')

    plt.show()


x = [1, 2, 3]
y = [1, 2, 3]

w = -20
for i in range(10):
    g = gradient_descent(x, y, w)
    w -= 0.1 * g
    print(i, w)

# 문제
# w를 1.0으로 만들 수 있는 두가지 방법

# 1. range 를 키운다
w = 20
for i in range(100):
    g = gradient_descent(x, y, w)
    w -= 0.1 * g
    print(i, w)

# 2. learning rate 를 키워본다
w = 20
for i in range(10):
    g = gradient_descent(x, y, w)
    w -= 0.215 * g
    print(i, w)

# 위에 w를 찾고자 하는거니까 w는 우리가 손댈 수 없고
# 반복하는 횟수, 학습률(learning rate) 같이 우리가 손대야 하는 부분을 Hyper Parameter 라고 한다

w = 20
for i in range(100):
    c = cost(x, y, w)
    g = gradient_descent(x, y, w)
    w -= 0.1 * g
#    print(i, w)
    print(i, c)

# 위처럼 사실상 W의 값은 우리에가 큰 의미가 없기 때문에 c가 변하는 값을 출력하는게 더 중요하다

w = 20
for i in range(100):
    c = cost(x, y, w)
    g = gradient_descent(x, y, w)

#   early stopping : 기준을 C로 잡는것보다 C의 변화량 등 더 정확한 기준을 잡는게 중요하지만
#   어쨌든 중간에 나오게끔 하는 조건.
    if c < 1.0e-15:
        break

    w -= 0.1 * g
#    print(i, w)
    print(i, c)

# early stopping 을 하는 또 다른 이유:
# 오히려 training set 을 과하게 학습하다보면 test set 에 대하여 좋은 성적을 보이지 못하는 경우가 생긴다.
# 이를 over fitting 이라고 한다.
# 이러한 현상을 막기위해 early stopping 이 필요하다



