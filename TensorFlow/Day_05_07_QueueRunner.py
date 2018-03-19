# Day_05_07_QueueRunner.py
import tensorflow as tf


def runner_1():
    queue = tf.train.string_input_producer(['12', '34', '56'],
                                           shuffle=False)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    for i in range(20):
        value = sess.run(queue.dequeue())
        print(value, value.decode('utf-8'))

        if i % 3 == 2:
            print()

    coord.request_stop()
    coord.join(threads)


def runner_2():
    queue = tf.train.string_input_producer(['Data/Data/q1.txt',
                                            'Data/Data/q2.txt',
                                            'Data/Data/q2.txt'],
                                           shuffle=False)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    reader = tf.TextLineReader()
    key, value = reader.read(queue)
    print('  key :', sess.run(key))
    print('value :', sess.run(value))

    record_defaults = [[0.], [0.], [0.]]
    for i in range(20):
        x1, x2, x3 = tf.decode_csv(value, record_defaults)
        print(sess.run([x1, x2, x3]))

    coord.request_stop()
    coord.join(threads)


def runner_3():
    queue = tf.train.string_input_producer(['Data/Data/iris_softmax.csv'])

    reader= tf.TextLineReader()
    _, value = reader.read(queue)

    record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
    iris = tf.decode_csv(value, record_defaults)

    batch_size = 15
    batch_x, batch_y = tf.train.batch([iris[:-3], iris[-3:]], batch_size)

    # --------------------------------- #

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    w = tf.Variable(tf.zeros([5, 3]))

    # (120, 3) = (120, 5) x (5, 3)
    z = tf.matmul(x, w)
    hypothesis = tf.nn.softmax(z)

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z,
                                                     labels=y)
    cost = tf.reduce_mean(cost_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    sess.run(tf.global_variables_initializer())

    epochs = 10
    for i in range(epochs):
        count = 150 // batch_size
        total = 0
        for _ in range(count):
            xx, yy = sess.run([batch_x, batch_y])
            sess.run(train, feed_dict={x: xx, y: yy})
            total += sess.run(cost, feed_dict={x: xx, y: yy})

        print(i, total / count)

    coord.request_stop()
    coord.join(threads)

    sess.close()

# runner_1()
# runner_2()
runner_3()