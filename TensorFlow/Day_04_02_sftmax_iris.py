import tensorflow as tf
import numpy as np
from sklearn import model_selection

def train_split_test():
    iris = np.loadtxt('Data/Data/iris_softmax.csv', delimiter=',')

    # print(iris.shape)

    x_data = iris[:, :-3]
    y_data = iris[:, -3:]
    # print(x_data.shape, y_data.shape)

    # data 셔플해서 train set, test set 나눠주는 함수 ( 75: 25 )
    # x 자료와 y 자료를 나누어서 넘겨줘야함
    data = model_selection.train_test_split(x_data, y_data)
    x_train, x_test, y_train, y_test = data
    print(type(data))
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print('-'*50)

    # 실수 : 비율로 계산
    data = model_selection.train_test_split(x_data, y_data, train_size=0.7)
    x_train, x_test, y_train, y_test = data
    print(type(data))
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print('-'*50)

    # 양수 : 실 데이터 갯수
    data = model_selection.train_test_split(x_data, y_data, train_size=120)
    x_train, x_test, y_train, y_test = data
    print(type(data))
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print('-'*50)


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

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y_train)

cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss=cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train, {x: x_train})
    # print(i, sess.run(cost, {x: x_train}))

pred = sess.run(hypothesis, {x: x_test})

# print(pred)

# print(np.argmax(pred, axis=1))
# print(np.argmax(y_test, axis=1))

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
