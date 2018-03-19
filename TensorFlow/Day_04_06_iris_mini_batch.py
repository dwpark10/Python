import tensorflow as tf
import numpy as np
from sklearn import model_selection

# 문제
# 120개의 데이터로 학습하고 30개의 정확도를 예측해보기
iris = np.loadtxt('Data/Data/iris_softmax.csv', delimiter=',')

x_data = iris[:, :-3]
y_data = iris[:, -3:]

data = model_selection.train_test_split(x_data, y_data, train_size=120)
x_train, x_test, y_train, y_test = data

x = tf.placeholder(tf.float32, shape=[None, 5])
y = tf.placeholder(tf.float32, shape=[None, 3])
w = tf.Variable(tf.zeros([5, 3]))

z = tf.matmul(x, w)

hypothesis = tf.nn.softmax(z)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)

cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss=cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 10
batch_size = 5

indices = np.arange(len(x_train))

count = len(x_train) // batch_size
for i in range(epochs):
    total_cost = 0

    for j in range(count):
        start = j * batch_size
        end = start + batch_size

        # _, loss = sess.run([train, cost], {x: x_train[start:end], y: y_train[start:end]})
        slice = indices[start:end]

        x_data = x_train[slice]
        y_data = y_train[slice]

        _, loss = sess.run([train, cost], {x: x_data, y: y_data})
        total_cost += loss

    np.random.shuffle(indices)
    print(indices[:10])
    print(i, total_cost/count)

pred = sess.run(hypothesis, {x: x_test})

index = np.argmax(pred, axis=1)
index_r = np.argmax(y_test, axis=1)

count = 0
for i in range(30):
    if index[i] == index_r[i]:
        count += 1

print((count/30) * 100, '%', sep='')

# spices = np.array(['setosa', 'versicolor', 'virginica'])
#
# print(spices[index])

sess.close()
