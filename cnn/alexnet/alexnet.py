
"""
Tensorflow implementation of AlexNet
"""
import tensorflow as tf
from wrapper import (kernel_layer, bias_layer, constant, norm,
                     max_pool, placeholder, conv_layer, matmul,
                     get_middle_shape)

class AlxNet(object):
    def __init__(self, n_classes, keep_prob=0.5, train=False):
        """
        The middle shape in AlexNet is designed to be 4096, however,
        we are using larger images so the fully connected layers
        need to be modified
        """
        self.input_data = placeholder("input_data")
        self.keep_prob = constant(keep_prob, "Dropout")
        self.train = train

        self.n_classes = n_classes

        with tf.variable_scope("pool1"):
            self.weights1 = kernel_layer([11, 11, 3, 48], 'weights')
            self.bias1 = bias_layer([48], 'bias', 1.0)
            self.conv1 = conv_layer(self.input_data, self.weights1, self.bias1, [1, 4, 4, 1])
            self.norm1 = norm(self.conv1, 'norm')
            self.pool1 = max_pool(self.norm1, 'pool', [1, 3, 3, 1], [1, 2, 2, 1])

        with tf.variable_scope("pool2"):
            self.weights2 = kernel_layer([5, 5, 48, 128], 'weights')
            self.bias2 = bias_layer([128], 'bias', 1.0)
            self.conv2 = conv_layer(self.pool1, self.weights2, self.bias2)
            self.norm2 = norm(self.conv2, "norm2")
            self.pool2 = max_pool(self.norm2, 'pool2', [1, 3, 3, 1], [1, 2, 2, 1])

        with tf.variable_scope("pool3"):
            with tf.variable_scope('conv3'):
                self.weights3 = kernel_layer([3, 3, 128, 192], 'weights')
                self.bias3 = bias_layer([192], 'bias')
                self.conv3 = conv_layer(self.pool2, self.weights3, self.bias3)
            with tf.variable_scope("conv4"):
                self.weights4 = kernel_layer([3, 3, 192, 192], 'weights')
                self.bias4 = bias_layer([192], 'bias', 1.0)
                self.conv4 = conv_layer(self.conv3, self.weights4, self.bias4)
            with tf.variable_scope("conv5"):
                self.weights5 = kernel_layer([3, 3, 192, 128], 'weights')
                self.bias5 = bias_layer([128], 'bias', 1.0)
                self.conv5 = conv_layer(self.conv4, self.weights5, self.bias5)

            self.pool5 = max_pool(self.conv5, 'pool5', [1, 3, 3, 1], [1, 2, 2, 1])

        middle_shape = get_middle_shape(self.pool5)
        with tf.variable_scope("fc6"):
            self.reshape5 = tf.reshape(self.pool5, [-1, middle_shape])
            self.weights6 = kernel_layer([middle_shape, middle_shape], 'weights')
            self.bias6 = bias_layer([middle_shape], 'bias', 1.0)
            self.fc6 = matmul(self.reshape5, self.weights6, self.bias6)
            if self.train is True:
                self.fc6 = tf.nn.dropout(self.fc6, self.keep_prob)

        with tf.variable_scope("fc7"):
            self.weights7 = kernel_layer([middle_shape, middle_shape], 'weights')
            self.bias7 = bias_layer([middle_shape], 'bias', 1.0)
            self.fc7 = matmul(self.fc6, self.weights7, self.bias7)

            if self.train is True:
                self.fc7 = tf.nn.dropout(self.fc7, self.keep_prob)

        with tf.variable_scope("logits"):
            self.weights8 = kernel_layer([middle_shape, self.n_classes], 'weights')
            self.bias8 = bias_layer([self.n_classes], 'bias', 1.0)
            self.logits = matmul(self.fc7, self.weights8, self.bias8)

        self.softmax = tf.nn.softmax(self.logits, 'softmax')

