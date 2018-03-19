import tensorflow as tf

def save_model_1():
    xx = [1, 2, 3]
    y = [1, 2, 3]

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_normal([1], -1, 1))
    b = tf.Variable(tf.random_normal([1], -1, 1))

    hypothesis = w * x + b

    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for i in range(101):
        sess.run(train, {x: xx})
        print(i, sess.run(cost, {x: xx}))

        if i % 10 == 0:
            saver.save(sess, 'Model/second', global_step=i)

    new_xx = [5, 7]
    print(sess.run(hypothesis, {x: new_xx}))

    # saver.save(sess, 'Model/first')

    sess.close()


def restore_model_1():
    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_normal([1], -1, 1))
    b = tf.Variable(tf.random_normal([1], -1, 1))

    hypothesis = w * x + b

    sess = tf.Session()

    latest = tf.train.latest_checkpoint('Model')

    saver = tf.train.Saver()
    saver.restore(sess, latest)

    new_xx = [5, 7]
    print(sess.run(hypothesis, {x: new_xx}))

    sess.close()

def restore_model_1_2():
    w = tf.Variable(tf.random_normal([1], -1, 1))
    b = tf.Variable(tf.random_normal([1], -1, 1))

    sess = tf.Session()

    latest = tf.train.latest_checkpoint('Model')

    saver = tf.train.Saver()
    saver.restore(sess, latest)

    new_xx = [5, 7]
    hypothesis = w * new_xx + b
    print(sess.run(hypothesis))

    sess.close()


# save_model_1()
# restore_model_1()
# restore_model_1_2()

############################################################################

def save_model_2():
    xx = [1, 2, 3]
    y = [1, 2, 3]

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_normal([1], -1, 1), name='weight')
    b = tf.Variable(tf.random_normal([1], -1, 1), name='bias')

    hypothesis = w * x + b

    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for i in range(101):
        sess.run(train, {x: xx})
        print(i, sess.run(cost, {x: xx}))

        if i % 10 == 0:
            saver.save(sess, 'Model/third', global_step=i)

    new_xx = [5, 7]
    print(sess.run(hypothesis, {x: new_xx}))

    sess.close()


def restore_model_2():
    w = tf.Variable([0.], name='weight')
    b = tf.Variable([0.], name='bias')

    sess = tf.Session()

    latest = tf.train.latest_checkpoint('Model')

    saver = tf.train.Saver()
    saver.restore(sess, latest)

    new_xx = [5, 7]
    hypothesis = w * new_xx + b
    print(sess.run(hypothesis))

    sess.close()

# save_model_2()
# restore_model_2()



# pickle => python lib