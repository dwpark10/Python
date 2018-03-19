for i in range(0, 5, 2):
    print(i, end=" ")
print()

for i in range(0, 10):
    print(i, end=" ")
print()

for i in range(5):
    print(i, end=" ")
print()

for i in reversed(range(5)):
    print(i, end=" ")
print()

for i, v in enumerate(range(5)):
    print(i, v ** 2)
print()

# collectin : list, tuple, dict
#              []     ()    {}


# list
a = [1, 3 ,5]
print(a, len(a))
print(a[0], a[1], a[2])
print(a[len(a)-1], a[-1])

a.append(7)
a.extend([9])   # 안에 list 객체가 들어가야함
a += [11]

print(a)

for i in range(len(a)):
    print(i, end=' ')
print()

for i in range(len(a)):
    print(a[i], end=' ')
print()

for i in a:
    print(i, end= ' ')
print()

for i in reversed(range(len(a))):
    print(a[i], end=' ')
print()

for i in reversed(a):
    print(i, end=' ')
print()


# tuple
a = (1, 3, 5)
print(a)

# a[1] = 7
# a.append(7)
# 과 같이 tuple 은 내부 데이터를 변경할 수 없다 ( list 상수 버전 )

t1 = (3, 8)
print(t1, type(t1))

t2 = 3, 8
print(t2, type(t2))
# 파이썬 문법적으로 괄호 없이 써도 튜플로 됨

t3, t4 = (3, 8)
print(t3, t4, type(t3), type(t4))

t3, t4 = t2
print(t3, t4, type(t3), type(t4))

t5 = t3, t4
print(t5, type(t5))

def f_3(k1, k2):
    return k1+k2, k1*k2

res = f_3(2, 4)
print(res, type(res))

res1, res2 = f_3(2, 4)
print(res1, res2, type(res1), type(res2))

res3, _ = f_3(3, 8)   # _ : place holder ( 데이터를 변수로 사용하지 않겠다, 무시하겠다 의미 )
print(res3, type(res3))

m = [1, 2, 3]
m1, m2, m3 = [1, 2, 3]
# list 도 괄호 벗겨져서 들어갈 수 있음

# dictionary

x = 123

d = {'name':'hoon', 'age':20, 3:4, x:456}
print(d)
print(d['name'], d['age'], d[3])
# 문자열 key 가 아니라도 만들 수 있다


d = dict(name='hoon', age=20, x=345)
print(d)
# d = dict(name='hoon', age=20, 3=4) 와 같이 int 형 key를 넣을 수 없다

# 위 두 dict 에서 x 가 동작하는 거를 보면 차이를 알 수 있음

print(d.keys())
print(d.items())

for k in d:
    print(k, d[k])
print()

for k, v in d.items():
    print(k, v)
print()

def f_4(*args):
    print(args, *args)
print()
# 가변인자

f_4()
f_4(1)
f_4(1, "hello")

a = [4, 7, 0]
print(a, *a)
print(*a, sep='\n')
# 함수 파라메터가 아닌 위와 같이 * 를 쓰면 print할때 []가 벗겨져서 나옴


