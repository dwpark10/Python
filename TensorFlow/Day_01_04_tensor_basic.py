import tensorflow as tf


def basic():
    a = tf.constant(3)
    b = tf.Variable(5)
    add = tf.add(a, b)
    # mul = tf.multiply(a, b)
    mul = a * b     # 연산자 오버로딩 ( 이 연산은 tensor * tensor 이다 )
    ad = a + b


    print(a)
    print(b)

    sess = tf.Session()
    #init = tf.global_variables_initializer()
    #sess.run(init)
    sess.run(tf.global_variables_initializer())

    print(sess.run(a), sess.run(b))
    print(sess.run(add))
    print(sess.run(ad))

    print(sess.run(mul))

    sess.close()

def place_holder():
    aa = tf.placeholder(tf.int32)
    bb = tf.placeholder(tf.int32)

    #add = tf.add(3, 5) -> 문제점이 있다. 3과 5만 더하는 연산으로 쓰이게 된다. 그래서 placeholder 를 쓰는데...

    add = tf.add(aa, bb)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #feed = {aa: 2, bb: 7}
    #print(sess.run(add, feed_dict=feed))
    print(sess.run(add, feed_dict={aa: 2, bb: 7}))
    print(sess.run(add, {aa: 2, bb: 7}))
    sess.close()