import numpy as np

def slicing():
    a = range(10)
    print(a)

    a = list(range(10))
    print(a)

    print(a[1], a[9])

    print(a[3:7])
    print(a[3:9:2])

    print(a[0:len(a)//2])
    print(a[len(a)//2:len(a)])
    print(a[0::2])
    print(a[1::2])

    # 아래의 두개의 차이를 잘 알아두자
    print(a[3:4], a[3])


    print(a[-1::-1])

a = np.array([1, 3, 5])
# numpy 의 array 를 출력해보면 data 사이에 , 가 없다
print(a)

# 일반적인 python index 문법을 모두 사용할 수 있다
print(a)
print(a[0], a[-1])
print(a[:-1])

a += 1      # broadcasting
print(a)
print(a ** 2)
print(a >= 4)
print(a[a >= 4])    # boolian 배열
print(np.sin(a))    # universal function
# print(a.sin())  이거는 error
print('-'*30)


b1 = np.arange(6)
b2 = np.arange(6).reshape(2,3)
print(b1)
print(b2)
print(b1.shape, b1.dtype)
print(b2.shape, b2.dtype)

print(b1.reshape(-1, 3))    # 어짜피 나누어 떨어질거니까 -1부분은 알아서 채워 라는 뜻
print(b1.reshape(3, -1))

# 문제
# b2를 1차원으로 reshape 해보세요
print(b2.reshape(-1))
# b2.reshape([6])
# np.reshape(b2, [6])
print('-'*30)


c = np.arange(3)
print(c)
print(c + c)
print(c * c)
print('-'*30)


d = np.arange(12).reshape(-1, 3)
print(d)
print(np.sum(d))    # 전체 합계
print(d.sum(axis=0))    # 세로 합
print(d.sum(axis=1))    # 가로 합
print(np.mean(d.sum(axis=0)))
print(np.mean(d.sum(axis=1)))

print(np.argmax(d)) # 가장 큰 값의 '위치(index)' 를 찾는다
print(np.argmax(d, axis=0)) # 가장 큰 값의 '위치(index)' 를 찾는다
print(np.argmax(d, axis=1)) # 가장 큰 값의 '위치(index)' 를 찾는다
print('-'*30)


e = np.arange(12).reshape(-1, 3)
print(e[0])
print(e[-1])
print(e[:2])
print(e[-1:])
print(e[[0, -1]])   # index 배열

idx = [-1, 1, -1]
print(e[idx])
print('-'*30)


f = np.arange(12).reshape(-1, 3)
print(f[0][0], f[-1][-1])   # numpy 에서 이 연산은 앞쪽 배열을 한번 가져온 후 다시 배열을 가져오는, 두번 연산을 하기 때문에 아래와 같은 코드를 쓴다
print(f[0, 0], f[-1, -1])   # fancy indexing
print(f[:2, 0])
print(f[1:-1, 1:-1])
print('-'*30)


g1 = np.arange(3)
g2 = np.arange(3).reshape(1, -1)
g3 = np.arange(3).reshape(-1, 1)
g4 = np.arange(6).reshape(2, -1)

print(g1)
print(g2)
print(g3)
print(g4)


print(g1 + g1)
print(g1 + g2)
print(g1 + g3)
print(g2 + g3)

print(g1 + g4)


yy = [1, 2, 3, 4, 5]
y = np.array(yy, dtype=float)
print(type(y))
print(type(y[0]))