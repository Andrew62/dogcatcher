
"""
Tensorflow implementation of AlexNet
"""
import tensorflow as tf
from wrapper import kernel, bias, constant, norm, max_pool

class AlxNet(object):
    def __init__(self, n_classes, middle_shape=8192, keep_prob=0.5):
        """
        The middle shape in AlexNet is designed to be 4096, however,
        we are using larger images so the fully connected layers
        need to be modified
        """

        self.keep_prob = constant(keep_prob, "Dropout")

        self.middle_shape = middle_shape
        self.n_classes = n_classes

        self.layers = {
            1: kernel([11, 11, 3, 48], 'kernel_1'),
            2: kernel([5, 5, 48, 128], 'kernel_2'),
            3: kernel([3, 3, 128, 192], 'kernel_3'),
            4: kernel([3, 3, 192, 192], 'kernel_4'),
            5: kernel([3, 3, 192, 128], 'kernel_5'),
            6: kernel([self.middle_shape, self.middle_shape], 'layer_6'),
            7: kernel([self.middle_shape, self.middle_shape], 'layer_7'),
            8: kernel([self.middle_shape, self.n_classes], 'layer_8')

        }

        self.biases = {
            1: bias([48], 'biases_1', 0.0),
            2: bias([128], 'biases_2', 1.0),
            3: bias([192], 'biases_3', 0.0),
            4: bias([192], 'biases_4', 1.0),
            5: bias([128], 'biases_5', 1.0),
            6: bias([self.middle_shape], 'biases_6', 1.0),
            7: bias([self.middle_shape], 'biases_7', 1.0),
            8: bias([self.n_classes], 'biases_8', 1.0)
        }



    def predict(self, data, train=False):
        conv_1 = tf.nn.conv2d(data, self.layers[1], [1, 4, 4, 1],
                              padding='SAME', name='convolution_1')

        hidden_1 = tf.nn.relu(conv_1 + self.biases[1], name='ReLU_1')

        norm_1 = norm(hidden_1, 'norm_1')

        max_pool_1 = max_pool(norm_1, 'max_pool_1')

        conv_2 = tf.nn.conv2d(max_pool_1, self.layers[2], strides=[1, 1, 1, 1],
                              padding='SAME', name='conv_2')

        hidden_2 = tf.nn.relu(conv_2 + self.biases[2], name='ReLU_2')

        norm_2 = norm(hidden_2, "norm_2")

        max_pool_2 = max_pool(norm_2, 'max_pool_2')

        conv_3 = tf.nn.conv2d(max_pool_2, self.layers[3], strides=[1, 1, 1, 1],
                              padding='SAME', name='conv_3')

        hidden_3 = tf.nn.relu(conv_3 + self.biases[3], name='relu_3')

        conv_4 = tf.nn.conv2d(hidden_3, self.layers[4], strides=[1, 1, 1, 1],
                              padding='SAME', name='conv_4')

        hidden_4 = tf.nn.relu(conv_4 + self.biases[4], name='hidden_4')

        conv_5 = tf.nn.conv2d(hidden_4, self.layers[5], strides=[1, 1, 1, 1],
                              padding='SAME', name='conv_5')

        max_pool_5 = max_pool(conv_5, 'max_pool_5')

        reshape_max_pool_5 = tf.reshape(max_pool_5, [-1, self.middle_shape])

        matmul_6 = tf.matmul(reshape_max_pool_5, self.layers[6])

        hidden_6 = tf.nn.relu(matmul_6 + self.biases[6], name='hidden_6')

        if train is True:
            hidden_6 = tf.nn.dropout(hidden_6, self.keep_prob)

        matmul_7 = tf.matmul(hidden_6, self.layers[7])

        hidden_7 = tf.nn.relu(matmul_7 + self.biases[7], name='hidden_7')

        if train is True:
            hidden_7 = tf.nn.dropout(hidden_7, self.keep_prob)

        matmul_8 = tf.matmul(hidden_7, self.layers[8])

        return tf.nn.relu(matmul_8 + self.biases[8], name="logits")