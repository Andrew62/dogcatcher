
"""
A tensorflow implementation of VGG19. Structure was pulled
from the paper. Renamed layers for finetuning
"""

import tensorflow as tf
from operator import mul
from layers import variable_summaries

class VGG16_D(object):

    def __init__(self, n_classes=1000, keep_prob=0.5, train=False):

        self.keep_prob = tf.constant(keep_prob, name="Dropout", dtype=tf.float32)

        self.input_data = tf.placeholder(dtype=tf.float32, name='input_data', shape=[None, 224, 224, 3])

        self.layer_names = []

        mean = tf.constant([125.974950491, 121.990847064, 102.991749558],
                           dtype=tf.float32, name='img_mean')

        self.mean_subtract = self.input_data - mean

        name = 'conv1_1'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights1 = tf.get_variable('weights',[3, 3, 3, 64], initializer=tf.random_normal_initializer(stddev=1e-2),
                                            trainable=False)
            self.bias1 = tf.get_variable("biases", [64], initializer=tf.constant_initializer(0.0),
                                         trainable=False)
            self.convolve1 = tf.nn.conv2d(self.mean_subtract, self.weights1, [1, 1, 1, 1], padding="SAME")
            self.conv1 = tf.nn.relu(tf.nn.bias_add(self.convolve1, self.bias1))

        name = 'conv1_2'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights2 = tf.get_variable('weights', [3, 3, 64, 64],
                                            initializer=tf.random_normal_initializer(stddev=1e-2),
                                            trainable=False)
            self.bias2 = tf.get_variable("biases", [64], initializer=tf.constant_initializer(0.0),
                                         trainable=False)
            self.convolve2 = tf.nn.conv2d(self.conv1, self.weights2, [1, 1, 1, 1], padding="SAME")
            self.conv2 = tf.nn.relu(tf.nn.bias_add(self.convolve2, self.bias2))

        self.pool1 = tf.nn.max_pool(self.conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool1')
        variable_summaries(self.pool1, 'pool1')
        self.layer_names.append('pool1')

        name = 'conv2_1'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights3 = tf.get_variable('weights', [3, 3, 64, 128],
                                            initializer=tf.random_normal_initializer(stddev=1e-2),
                                            trainable=False)
            self.bias3 = tf.get_variable("biases", [128], initializer=tf.constant_initializer(0.0),
                                         trainable=False)
            self.convolve3 = tf.nn.conv2d(self.pool1, self.weights3, [1, 1, 1, 1], padding="SAME")
            self.conv3 = tf.nn.relu(tf.nn.bias_add(self.convolve3, self.bias3))

        name = 'conv2_2'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights4 = tf.get_variable('weights', [3, 3, 128, 128],
                                            initializer=tf.random_normal_initializer(stddev=1e-2),
                                            trainable=False)
            self.bias4 = tf.get_variable("biases", [128], initializer=tf.constant_initializer(0.0),
                                         trainable=False)
            self.convolve4 = tf.nn.conv2d(self.conv3, self.weights4, [1, 1, 1, 1], padding="SAME")
            self.conv4 = tf.nn.relu(tf.nn.bias_add(self.convolve4, self.bias4))
        self.pool2 = tf.nn.max_pool(self.conv4, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool2')
        self.layer_names.append("pool2")
        variable_summaries(self.pool2, 'pool2')

        name = 'conv3_1'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights5 = tf.get_variable('weights', [3, 3, 128, 256],
                                            initializer=tf.random_normal_initializer(stddev=1e-2),
                                            trainable=False)
            self.bias5 = tf.get_variable("biases", [256], initializer=tf.constant_initializer(0.0),
                                         trainable=False)
            self.convolve5 = tf.nn.conv2d(self.pool2, self.weights5, [1, 1, 1, 1], padding="SAME")
            self.conv5 = tf.nn.relu(tf.nn.bias_add(self.convolve5, self.bias5))

        name = 'conv3_2'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights6 = tf.get_variable('weights', [3, 3, 256, 256],
                                            initializer=tf.random_normal_initializer(stddev=1e-2),
                                            trainable=False)
            self.bias6 = tf.get_variable("biases", [256], initializer=tf.constant_initializer(0.0),
                                         trainable=False)
            self.convolve6 = tf.nn.conv2d(self.conv5, self.weights6, [1, 1, 1, 1], padding="SAME")
            self.conv6 = tf.nn.relu(tf.nn.bias_add(self.convolve6, self.bias6))

        name = 'conv3_3'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights7 = tf.get_variable('weights', [3, 3, 256, 256],
                                            initializer=tf.random_normal_initializer(stddev=1e-2),
                                            trainable=False)
            self.bias7 = tf.get_variable("biases", [256], initializer=tf.constant_initializer(0.0),
                                         trainable=False)
            self.convolve7 = tf.nn.conv2d(self.conv6, self.weights7, [1, 1, 1, 1], padding="SAME")
            self.conv7 = tf.nn.relu(tf.nn.bias_add(self.convolve7, self.bias7))

        self.pool3 = tf.nn.max_pool(self.conv7, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool3')
        self.layer_names.append('pool3')
        variable_summaries(self.pool3, 'pool3')

        name = 'conv4_1'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights8 = tf.get_variable('weights', [3, 3, 256, 512],
                                            initializer=tf.random_normal_initializer(stddev=1e-2),
                                            trainable=False)
            self.bias8 = tf.get_variable("biases", [512], initializer=tf.constant_initializer(0.0),
                                         trainable=False)
            self.convolve8 = tf.nn.conv2d(self.pool3, self.weights8, [1, 1, 1, 1], padding="SAME")
            self.conv8 = tf.nn.relu(tf.nn.bias_add(self.convolve8, self.bias8))

        name = 'conv4_2'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights9 = tf.get_variable('weights', [3, 3, 512, 512],
                                            initializer=tf.random_normal_initializer(stddev=1e-2),
                                            trainable=False)
            self.bias9 = tf.get_variable("biases", [512], initializer=tf.constant_initializer(0.0),
                                         trainable=False)
            self.convolve9 = tf.nn.conv2d(self.conv8, self.weights9, [1, 1, 1, 1], padding="SAME")
            self.conv9 = tf.nn.relu(tf.nn.bias_add(self.convolve9, self.bias9))

        name = 'conv4_3'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights10 = tf.get_variable('weights', [3, 3, 512, 512],
                                             initializer=tf.random_normal_initializer(stddev=1e-2),
                                             trainable=False)
            self.bias10 = tf.get_variable("biases", [512], initializer=tf.constant_initializer(0.0),
                                          trainable=False)
            self.convolve10 = tf.nn.conv2d(self.conv9, self.weights10, [1, 1, 1, 1], padding="SAME")
            self.conv10 = tf.nn.relu(tf.nn.bias_add(self.convolve10, self.bias10))
        self.pool4 = tf.nn.max_pool(self.conv10, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name="pool4")
        self.layer_names.append('pool4')
        variable_summaries(self.pool4, 'pool4')

        name = 'conv5_1'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights11 = tf.get_variable('weights', [3, 3, 512, 512],
                                             initializer=tf.random_normal_initializer(stddev=1e-2),
                                             trainable=False)
            self.bias11 = tf.get_variable("biases", [512], initializer=tf.constant_initializer(0.0),
                                          trainable=False)
            self.convolve11 = tf.nn.conv2d(self.pool4, self.weights11, [1, 1, 1, 1], padding="SAME")
            self.conv11 = tf.nn.relu(tf.nn.bias_add(self.convolve11, self.bias11))

        name = 'conv5_2'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights12 = tf.get_variable('weights', [3, 3, 512, 512],
                                             initializer=tf.random_normal_initializer(stddev=1e-2),
                                             trainable=False)
            self.bias12 = tf.get_variable("biases", [512], initializer=tf.constant_initializer(0.0),
                                          trainable=False)
            self.convolve12 = tf.nn.conv2d(self.conv11, self.weights12, [1, 1, 1, 1], padding="SAME")
            self.conv12 = tf.nn.relu(tf.nn.bias_add(self.convolve12, self.bias12))

        name = 'conv5_3'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights13 = tf.get_variable('weights', [3, 3, 512, 512],
                                             initializer=tf.random_normal_initializer(stddev=1e-2),
                                             trainable=False)
            self.bias13 = tf.get_variable("biases", [512], initializer=tf.constant_initializer(0.0),
                                          trainable=False)
            self.convolve13 = tf.nn.conv2d(self.conv12, self.weights13, [1, 1, 1, 1], padding="SAME")
            self.conv13 = tf.nn.relu(tf.nn.bias_add(self.convolve13, self.bias13))
        self.pool5 = tf.nn.max_pool(self.conv13, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name="pool5")
        variable_summaries(self.pool5, 'pool5')
        middle_shape = reduce(mul, self.pool5.get_shape().as_list()[1:], 1)
        self.reshape5 = tf.reshape(self.pool5, [-1, middle_shape])
        self.layer_names.append('pool5')


        name = 'fc6'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights14 = tf.get_variable('weights', [middle_shape, 4096],
                                             initializer=tf.random_normal_initializer(stddev=1e-2),
                                             trainable=False)
            self.bias14 = tf.get_variable("biases", [4096], initializer=tf.constant_initializer(0.0),
                                          trainable=False)
            self.fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.reshape5, self.weights14), self.bias14))
            if train is True:
                self.fc6 = tf.nn.dropout(self.fc6, self.keep_prob)

            variable_summaries(self.fc6, 'fc6')

        name = 'fc7'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights15 = tf.get_variable('weights', [4096, 4096],
                                             initializer=tf.random_normal_initializer(stddev=1e-2))
            self.bias15 = tf.get_variable("biases", [4096], initializer=tf.constant_initializer(0.0))
            self.fc7 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.fc6, self.weights15), self.bias15))
            if train is True:
                self.fc7 = tf.nn.dropout(self.fc7, self.keep_prob)
            variable_summaries(self.fc7, 'fc7')

        name = 'fc8'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            self.weights16 = tf.get_variable('weights', [4096, n_classes],
                                             initializer=tf.random_normal_initializer(stddev=1e-2))
            self.bias16 = tf.get_variable("biases", [n_classes], initializer=tf.constant_initializer(0.0))

        self.logits = tf.nn.bias_add(tf.matmul(self.fc7, self.weights16), self.bias16)
        self.softmax = tf.nn.softmax(self.logits, 'softmax')


if __name__ == "__main__":
    vgg = VGG16_D(252)
    print vgg.batch_norm.get_shape()
    for i in range(8):
        try:
            print getattr(vgg, 'weights{}'.format(i + 1)).get_shape()
        except:
            pass
