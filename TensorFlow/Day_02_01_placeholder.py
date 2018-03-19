import tensorflow as tf


# 문제: 구구단
def my_mul():
    x = tf.placeholder(tf.float32)
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    mul = x * y

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(sess.run(mul, {x: 5}))
    sess.close()

def new_mul(dan):
    left = tf.placeholder(tf.int32)
    right = tf.placeholder(tf.int32)

    multiply = left * right

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1, 10):
        print(sess.run(multiply, {left: dan, right: i}))

    sess.close()

# 문제 : 불필요한 부분을 삭제하세요
def new_mul2(dan):
#    left = tf.placeholder(tf.int32)
    right = tf.placeholder(tf.int32)

    multiply = dan * right

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(1, 10):
        result = sess.run(multiply, {right: i})
        print('{} x {} = {}'.format(dan, i, result))
    sess.close()

new_mul2(7)