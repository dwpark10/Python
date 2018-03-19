import numpy as np
import tensorflow as tf
from sklearn import model_selection, preprocessing, datasets, linear_model


def foo_1():
    iris = datasets.load_iris()
    print(iris.keys())
    # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

    print(iris['target_names'])
    print(iris.feature_names)

    print(type(iris.data))
    print(iris.data.shape)
    print(iris.target.shape)

    data = preprocessing.add_dummy_feature(iris.data)
    print(data.shape)
    print('-'*50)

    print(data[:3])
    print(iris.target[:5])
    print(iris.target)

    data = model_selection.train_test_split(data, iris.target, random_state=0)
    x_train, x_test, y_train, y_test = data

    regression = linear_model.LogisticRegression()
    regression.fit(x_train, y_train)

    print(regression.score(x_train, y_train))
    print(regression.score(x_test, y_test))


iris = datasets.load_iris()

data = preprocessing.add_dummy_feature(iris.data)
target = preprocessing.LabelBinarizer().fit_transform(iris.target)

print(target[:3])

data = model_selection.train_test_split(data, target, random_state=0)
x_total, x_test, y_total, y_test = data

data = model_selection.train_test_split(x_total, y_total, random_state=0)
x_train, x_valid, y_train, y_valid = data

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

w = tf.Variable(tf.zeros([5, 3]))

z = tf.matmul(x, w)

hypothesis = tf.nn.softmax(z)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)

cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss=cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


best_acc, best_lr = 0., 0.
for rate in [0.1, 0.01, 0.001]:
    for i in range(10):
        sess.run(train, {x: x_train,
                         y: y_train,
                         lr: rate})
        # print(i, sess.run(cost, {x: x_train}))

    y_hat = sess.run(hypothesis, {x: x_valid})

    y_hat_bool = np.argmax(y_hat, axis=1)
    y_valid_bool = np.argmax(y_valid, axis=1)

    equals = (y_hat_bool == y_valid_bool)
    acc = np.mean(equals)
    print('accuracy : ', acc)

    if best_acc < acc:
        best_acc = acc
        best_lr = rate

print(best_acc, best_lr)


# best lr 을 찾았으니 이제 더 좋은 결과를 만들기 위해 한번더 학습한다
# 앞서 나누었던 data set이 아닌 전체 데이터인 total data를 사용하면
# data set 의 절대적인 양이 많아지기 때문에 더 좋은 결과를 낼 수 있다.
for i in range(10):
    sess.run(train, {x: x_total,
                     y: y_total,
                     lr: best_lr})
    # print(i, sess.run(cost, {x: x_train}))

y_hat = sess.run(hypothesis, {x: x_test})

y_hat_bool = np.argmax(y_hat, axis=1)
y_test_bool = np.argmax(y_test, axis=1)

equals = (y_hat_bool == y_test_bool)
acc = np.mean(equals)
print('accuracy : ', acc)

sess.close()