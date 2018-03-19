import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python import SKCompat

iris = np.loadtxt('Data/Data/iris_softmax.csv', delimiter=',', dtype=np.float32)

x = iris[:, :-3]
y = iris[:, -3:]
y = np.argmax(y, axis=1)

feature_columns = [tf.feature_column.numeric_column('', shape=[5])]

clf = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 5], feature_columns=feature_columns, n_classes=3)

clf = SKCompat(clf)

clf.fit(x=x, y=y, max_steps=1000)

print(clf.score(x=x, y=y))