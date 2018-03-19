# Day_05_03_xavier.py
import tensorflow as tf
import numpy as np


def compute_fans(shape):
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    elif len(shape) > 2:
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim

        fan_in  = receptive_field_size * shape[-2]
        fan_out = receptive_field_size * shape[-1]

    return fan_in, fan_out


def test_xavier(shape, count=10):
    fan_in, fan_out = compute_fans(shape)
    limit = np.sqrt(6 / (fan_in + fan_out))

    print('fan-in  :', fan_in)
    print('fan-out :', fan_out)
    print('limit   :', -limit, limit)
    print('-' * 50)

    for i in range(count):
        v = tf.get_variable('v' + str(i), shape=shape)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        t = sess.run(v)
        sess.close()

        print(t)

        for item in t.reshape(-1):
            # assert -limit <= item <= limit
            if item < -limit or item > limit:
                print('--------- out of range.', item)


# test_xavier([5])
test_xavier([3, 5])
# test_xavier([3, 5, 4])
