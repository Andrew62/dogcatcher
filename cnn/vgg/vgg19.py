
"""
A tensorflow implementation of VGG19. Structure was pulled
from the paper.
"""

import tensorflow as tf

class VGG19(object):
    def __init__(self, n_classes=252):

        self.input_data = tf.placeholder(dtype=tf.float32, name='input_data', shape=(10,224,224,3))
        self.middle_shape = 25088

        with tf.variable_scope('batch_norm'):
            mean, var = tf.nn.moments(self.input_data, axes=[0, 1, 2])
            self.batch_norm = tf.nn.batch_normalization(self.input_data, mean, var, offset=None, scale=None,
                                                        variance_epsilon=1e4)

        with tf.variable_scope("group1"):
            with tf.variable_scope("conv1"):
                self.weights1 = tf.get_variable('weights',[3, 3, 3, 64], initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias1 = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
                self.convolve1 = tf.nn.conv2d(self.batch_norm, self.weights1, [1, 1, 1, 1], padding="SAME")
                self.conv1 = tf.nn.elu(tf.nn.bias_add(self.convolve1, self.bias1))
            with tf.variable_scope("conv2"):
                self.weights2= tf.get_variable('weights', [3, 3, 64, 64],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias2 = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
                self.convolve2 = tf.nn.conv2d(self.conv1, self.weights2, [1, 1, 1, 1], padding="SAME")
                self.conv2 = tf.nn.elu(tf.nn.bias_add(self.convolve2, self.bias2))

            self.pool1 = tf.nn.max_pool(self.conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool1')

        with tf.variable_scope("group2"):
            with tf.variable_scope("conv3"):
                self.weights3 = tf.get_variable('weights', [3, 3, 64, 128],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias3 = tf.get_variable('bias', [128], initializer=tf.constant_initializer(0.0))
                self.convolve3 = tf.nn.conv2d(self.pool1, self.weights3, [1, 1, 1, 1], padding="SAME")
                self.conv3 = tf.nn.elu(tf.nn.bias_add(self.convolve3, self.bias3))
            with tf.variable_scope("conv4"):
                self.weights4 = tf.get_variable('weights', [3, 3, 128, 128],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias4 = tf.get_variable('bias', [128], initializer=tf.constant_initializer(0.0))
                self.convolve4 = tf.nn.conv2d(self.conv3, self.weights4, [1, 1, 1, 1], padding="SAME")
                self.conv4 = tf.nn.elu(tf.nn.bias_add(self.convolve4, self.bias4))
            self.pool2 = tf.nn.max_pool(self.conv4, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool2')

        with tf.variable_scope("group3"):
            with tf.variable_scope("conv5"):
                self.weights5 = tf.get_variable('weights', [3, 3, 128, 256],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias5 = tf.get_variable('bias', [256], initializer=tf.constant_initializer(0.0))
                self.convolve5 = tf.nn.conv2d(self.pool2, self.weights5, [1, 1, 1, 1], padding="SAME")
                self.conv5 = tf.nn.elu(tf.nn.bias_add(self.convolve5, self.bias5))
            with tf.variable_scope("conv6"):
                self.weights6 = tf.get_variable('weights', [3, 3, 256, 256],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias6 = tf.get_variable('bias', [256], initializer=tf.constant_initializer(0.0))
                self.convolve6 = tf.nn.conv2d(self.conv5, self.weights6, [1, 1, 1, 1], padding="SAME")
                self.conv6 = tf.nn.elu(tf.nn.bias_add(self.convolve6, self.bias6))
            with tf.variable_scope("conv7"):
                self.weights7 = tf.get_variable('weights', [3, 3, 256, 256],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias7 = tf.get_variable('bias', [256], initializer=tf.constant_initializer(0.0))
                self.convolve7 = tf.nn.conv2d(self.conv6, self.weights7, [1, 1, 1, 1], padding="SAME")
                self.conv7 = tf.nn.elu(tf.nn.bias_add(self.convolve7, self.bias7))
            with tf.variable_scope("conv8"):
                self.weights8 = tf.get_variable('weights', [3, 3, 256, 256],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias8 = tf.get_variable('bias', [256], initializer=tf.constant_initializer(0.0))
                self.convolve8 = tf.nn.conv2d(self.conv7, self.weights8, [1, 1, 1, 1], padding="SAME")
                self.conv8 = tf.nn.elu(tf.nn.bias_add(self.convolve8, self.bias8))
            self.pool3 = tf.nn.max_pool(self.conv8, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool3')

        with tf.variable_scope("group4"):
            with tf.variable_scope("conv9"):
                self.weights9 = tf.get_variable('weights', [3, 3, 256, 512],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias9 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve9 = tf.nn.conv2d(self.pool3, self.weights9, [1, 1, 1, 1], padding="SAME")
                self.conv9 = tf.nn.elu(tf.nn.bias_add(self.convolve9, self.bias9))
            with tf.variable_scope("conv10"):
                self.weights10 = tf.get_variable('weights', [3, 3, 512, 512],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias10 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve10 = tf.nn.conv2d(self.conv9, self.weights10, [1, 1, 1, 1], padding="SAME")
                self.conv10 = tf.nn.elu(tf.nn.bias_add(self.convolve10, self.bias10))
            with tf.variable_scope("conv11"):
                self.weights11 = tf.get_variable('weights', [3, 3, 512, 512],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias11 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve11 = tf.nn.conv2d(self.conv10, self.weights11, [1, 1, 1, 1], padding="SAME")
                self.conv11 = tf.nn.elu(tf.nn.bias_add(self.convolve11, self.bias11))
            with tf.variable_scope("conv12"):
                self.weights12 = tf.get_variable('weights', [3, 3, 512, 512],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias12 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve12 = tf.nn.conv2d(self.conv11, self.weights12, [1, 1, 1, 1], padding="SAME")
                self.conv12 = tf.nn.elu(tf.nn.bias_add(self.convolve12, self.bias12))
            self.pool4 = tf.nn.max_pool(self.conv12, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name="pool4")

        with tf.variable_scope("group5"):
            with tf.variable_scope("conv13"):
                self.weights13 = tf.get_variable('weights', [3, 3, 512, 512],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias13 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve13 = tf.nn.conv2d(self.pool4, self.weights13, [1, 1, 1, 1], padding="SAME")
                self.conv13 = tf.nn.elu(tf.nn.bias_add(self.convolve13, self.bias13))

            with tf.variable_scope("conv14"):
                self.weights14 = tf.get_variable('weights', [3, 3, 512, 512],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias14 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve14 = tf.nn.conv2d(self.conv13, self.weights14, [1, 1, 1, 1], padding="SAME")
                self.conv14 = tf.nn.elu(tf.nn.bias_add(self.convolve14, self.bias14))
            with tf.variable_scope("conv15"):
                self.weights15 = tf.get_variable('weights', [3, 3, 512, 512],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias15 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve15 = tf.nn.conv2d(self.conv14, self.weights15, [1, 1, 1, 1], padding="SAME")
                self.conv15 = tf.nn.elu(tf.nn.bias_add(self.convolve15, self.bias15))
            with tf.variable_scope("conv16"):
                self.weights16 = tf.get_variable('weights', [3, 3, 512, 512],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias16 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve16 = tf.nn.conv2d(self.conv15, self.weights16, [1, 1, 1, 1], padding="SAME")
                self.conv16 = tf.nn.elu(tf.nn.bias_add(self.convolve16, self.bias16))
            self.pool5 = tf.nn.max_pool(self.conv16, [1, 2, 2, 1], [1, 2, 2, 1], "SAME", name="pool5")
            self.fc6 = tf.reshape(self.pool5, [-1, middle_shape], 'fc6')

        with tf.variable_scope("group6"):

            with tf.variable_scope("fc7"):
                self.weights17 = tf.get_variable('weights', [middle_shape, 4096],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias17 = tf.get_variable('bias', [4096], initializer=tf.constant_initializer(0.0))
                self.fc7 = tf.nn.elu(tf.nn.bias_add(tf.matmul(self.fc6, self.weights17), self.bias17))
            with tf.variable_scope("fc8"):
                self.weights18 = tf.get_variable('weights', [4096, 4096],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias18 = tf.get_variable('bias', [4096], initializer=tf.constant_initializer(0.0))
                self.fc8 = tf.nn.elu(tf.nn.bias_add(tf.matmul(self.fc7, self.weights18), self.bias18))
            with tf.variable_scope("logits"):
                self.weights19 = tf.get_variable('weights', [4096, n_classes],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias19 = tf.get_variable('bias', [n_classes], initializer=tf.constant_initializer(0.0))
                self.logits = tf.nn.bias_add(tf.matmul(self.fc8, self.weights19), self.bias19)

        self.softmax = tf.nn.softmax(self.logits, 'softmax')


if __name__ == "__main__":
    vgg = VGG19(10)
