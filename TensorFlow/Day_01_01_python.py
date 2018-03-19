# python 3.6.4 64bit 버전으로 ( 텐서플로우가 32비트 버전은 지원하지 않음
# editor : 파이참



# 실행 : ctrl + shift + f10
# alt + 1 , alt + 4
# alt + / : 주석
# 그냥 ctrl + c 누르면 해당 라인 복사


# 다중 치환
a, b = 3, 8
print(a, b)
a, b = b, a
print(a, b)

# 연산 : 산술, 관계, 논리
# 산술 연산자 : + - * / ** // %
print(a+b)
print(a-b)
print(a*b)
print(a/b)      # 실수 나눗셈
print(a**b)     # 지수
print(a//b)     # 정수 나눗셈 (몫)
print(a%b)      # 나머지

print("a" + 'b')
print("-" * 50)

# 관계 연산자 : > >= < <= == !=
print(a > b)
print(a >= b)
print(a < b)
print(a <= b)
print(a == b)
print(a != b)

age = 15
print(10 < age <= 20)   # python 에서는 이런 표현 가능

# 논리 연산자 : and or not
print(True and True)
print(True and False)
print(False and True)
print(False and False)

# 형변환
a = '3.14'
# print(a + 3.14) : error
print(a + str(3.14))
print(float(a) + 3.14)


# 제어문
a = 3

if a % 2 == 1:
    print("odd")
else:
    print("even")

if a < 0:
    print("negaive")
elif a > 0:
    print("positive")
else:
    print("zero")


# 함수
def f_1(a, b, c):
    print(a, b, c)

f_1(1, 2, 3)        # positional
f_1(a=1, b=2, c=3)  # keyword
f_1(c=3, b=2, a=1)  # 순서 상관없음

print(f_1(1, 2, 3)) # 함수의 기본 반환값 : None

def f_2(a, b, c):
    return a+b+c

print(f_2(1, 2, 3))
