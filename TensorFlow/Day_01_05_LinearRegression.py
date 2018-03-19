# numpi ? 이거 꼭 공부해야함

import tensorflow as tf

def regression_1():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(10.)
    b = tf.Variable(-5.)

    #hypothesis = tf.add(tf.multiply(w, x), b)
    hypothesis = w * x + b

    #cost = tf.reduce_mean(tf.square(hypothesis - y))
    cost = tf.reduce_mean((hypothesis - y) ** 2)


    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # i 값이 증가함에 따라 cost 값이 감소하는게 마지막까지 보인다면
    # range 를 늘려도 cost 가 더 감소 할 여지가 있다는 것.
    for i in range(10):
        print(sess.run(w), end=' ')
        print(sess.run(b), end=' ')
        print(sess.run(cost))
        sess.run(train)

    # 문제 : x 가 5, 7 일 때의 y값을 예측해보시오
    # print(sess.run(w*5+b))
    # print(sess.run(w*7+b))

    ww = sess.run(w)
    bb = sess.run(b)
    print('5: ', ww * 5 + bb)
    print('7: ', ww * 7 + bb)

    sess.close()


def temp():
    # 문제 : 위 함수를 placeholder 를 사용한 버전으로 수정하세요

    xx = [1, 2, 3]
    yy = [1, 2, 3]

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    w = tf.Variable(10.)
    b = tf.Variable(-5.)

    hypothesis = w * x + b

    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {x: xx, y: yy})
        print(i, sess.run(cost, {x: xx, y: yy}))

    new_xx = [5, 7]
    print(sess.run(hypothesis, {x: new_xx}))

    sess.close()

def regression_placeholder():
    # 위의 코드에서 x 값을 placeholder 로 준 이유는 x 값에 값을 바꿔가면서 넣을것이기 때문인데
    # y값은 처음 주어진 data 이외에는 바뀔 일이 없으므로
    # placeholder 로 줄 이유가 없다
    # 따라서 아래와 같이 고치는게 논리적으로 맞다 ( 위의 코드도 틀린 코드는 아님 )

    xx = [1, 2, 3]
    y = [1, 2, 3]

    x = tf.placeholder(tf.float32)
    #    y = tf.placeholder(tf.float32)

    w = tf.Variable(10.)
    b = tf.Variable(-5.)

    hypothesis = w * x + b

    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {x: xx})
        print(i, sess.run(cost, {x: xx}))

    new_xx = [5, 7]
    print(sess.run(hypothesis, {x: new_xx}))
    # 위처럼 모르는 값에 대한 예상 값을 찾고자 새로운 데이터를 집어넣었을 때 에러가 발생할 수 도 있다.
    # 하지만 아래처럼 이미 한번 들어갔던 data를 집어넣으면 거기서 만큼은 에러가 발생하면 안되기 때문에
    # 한번 확인하는? 용도로 사용할 수 있다.
    print(sess.run(hypothesis, {x: xx}))
    print(sess.run(hypothesis, {x: [1, 2, 3]}))
    sess.close()
