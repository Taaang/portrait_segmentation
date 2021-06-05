from tensorflow.python.ops.random_ops import truncated_normal
from tensorflow.keras.datasets import mnist

import tensorflow as tf

tf.compat.v1.disable_eager_execution()


# 生成卷积核
def weight_variable(shape):
    initial = truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 生成随机偏执
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积计算
def conv2d(input_data, conv_w):
    return tf.nn.conv2d(input_data, conv_w, strides=[1, 1, 1, 1], padding='SAME')


# 池化计算
def pool_2x2(input_data):
    return tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


(train_img, train_tag), (test_img, test_tag) = mnist.load_data()
train_data = tf.reshape(train_img, [-1, 28, 28, 1])


cnn_conv_1 = weight_variable([5, 5, 1, 6])
cnn_conv_1_bias = bias_variable([6])
cnn_conv_1_output = tf.nn.relu(tf.nn.bias_add(conv2d(train_data, cnn_conv_1) + cnn_conv_1_bias))
cnn_pool_1 = pool_2x2(cnn_conv_1_output)

print("End.")
