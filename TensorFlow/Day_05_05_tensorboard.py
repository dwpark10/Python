import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# python 문법
# class 안에서 @property 붙이는거 -> 함수이지만 사용할때는 ()를 붙이지 않고 변수를 쓰는것처럼 사용한다
# 한번 찾아보기

mnist = input_data.read_data_sets('mnist', one_hot=True)


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
# keep_rate = tf.placeholder(tf.float32)

##############################################################################################

with tf.name_scope('weights'):
    w = tf.Variable(tf.zeros([784, 10]))        # 784 = 28 * 28 / 이미지 사이즈
    b = tf.Variable(tf.zeros([10]))

with tf.name_scope('my_softmax'):
    z = tf.matmul(x, w) + b
    hx = tf.nn.softmax(z)

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)

    cost = tf.reduce_mean(cost_i)

##############################################################################################

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(cost)

# step 1.
tf.summary.scalar('cost', cost)


# step 2.
merged = tf.summary.merge_all()


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# step 3.
writer = tf.summary.FileWriter('board/mnist', sess.graph)


epochs, batch_size = 15, 100

count = mnist.train.num_examples // batch_size
for i in range(epochs):
    total_cost = 0
    for _ in range(count):
        xx, yy = mnist.train.next_batch(batch_size)

        _, c = sess.run([train, cost], {x: xx, y: yy})

        total_cost += c

        # step 4.
        summary = sess.run(merged, {x: xx, y: yy})
        writer.add_summary(summary, i)

    print(i, total_cost/count)


sess.close()