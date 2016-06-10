
"""
A tensorflow implementation of VGG19. Structure was pulled
from the paper.
"""

import tensorflow as tf
from .wrapper import kernel_layer, bias_layer, conv_layer, max_pool, matmul, placeholder


class VGG(object):
    def __init__(self, n_classes=252, middle_shape=100352):

        self.middle_shape = middle_shape

        self.input_data = placeholder('input_data')

        with tf.variable_scope("group1"):
            with tf.variable_scope("conv1"):
                self.weights1 = kernel_layer([3, 3, 3, 64], name='weights')
                self.bias1 = bias_layer([64], 'bias')
                self.conv1 = conv_layer(self.input_data, self.weights1, self.bias1)
            with tf.variable_scope("conv2"):
                self.weights2 = kernel_layer([3, 3, 64, 64], name='weights')
                self.bias2 = bias_layer([64], 'bias')
                self.conv2 = conv_layer(self.conv1, self.weights2, self.bias2)
            self.pool1 = max_pool(self.conv2, 'pool1')

        with tf.variable_scope("group2"):
            with tf.variable_scope("conv3"):
                self.weights3 = kernel_layer([3, 3, 64, 128], name='weights')
                self.bias3 = bias_layer([128], 'bias')
                self.conv3 = conv_layer(self.pool1, self.weights3, self.bias3)
            with tf.variable_scope("conv4"):
                self.weights4 = kernel_layer([3, 3, 128, 128], name='weights')
                self.bias4 = bias_layer([128], 'bias')
                self.conv4 = conv_layer(self.conv3, self.weights4, self.bias4)
            self.pool2 = max_pool(self.conv4, 'pool2')

        with tf.variable_scope("group3"):
            with tf.variable_scope("conv5"):
                self.weights5 = kernel_layer([3, 3, 128, 256], name='weights')
                self.bias5 = bias_layer([256], 'bias')
                self.conv5 = conv_layer(self.pool2, self.weights5, self.bias5)
            with tf.variable_scope("conv6"):
                self.weights6 = kernel_layer([3, 3, 256, 256], name='weights')
                self.bias6 = bias_layer([256], 'bias')
                self.conv6 = conv_layer(self.conv5, self.weights6, self.bias6)
            with tf.variable_scope("conv7"):
                self.weights7 = kernel_layer([3, 3, 256, 256], name='weights')
                self.bias7 = bias_layer([256], 'bias')
                self.conv7 = conv_layer(self.conv6, self.weights7, self.bias7)
            with tf.variable_scope("conv8"):
                self.weights8 = kernel_layer([3, 3, 256, 256], name='weights')
                self.bias8 = bias_layer([256], 'bias')
                self.conv8 = conv_layer(self.conv7, self.weights8, self.bias8)
            self.pool3 = max_pool(self.conv8, 'pool3')

        with tf.variable_scope("group4"):
            with tf.variable_scope("conv9"):
                self.weights9 = kernel_layer([3, 3, 256, 512], name='weights')
                self.bias9 = bias_layer([512], 'bias')
                self.conv9 = conv_layer(self.pool3, self.weights9, self.bias9)
            with tf.variable_scope("conv10"):
                self.weights10 = kernel_layer([3, 3, 512, 512], name='weights')
                self.bias10 = bias_layer([512], 'bias')
                self.conv10 = conv_layer(self.conv9, self.weights10, self.bias10)
            with tf.variable_scope("conv11"):
                self.weights11 = kernel_layer([3, 3, 512, 512], name='weights')
                self.bias11 = bias_layer([512], 'bias')
                self.conv11 = conv_layer(self.conv10, self.weights11, self.bias11)
            with tf.variable_scope("conv12"):
                self.weights12 = kernel_layer([3, 3, 512, 512], name='weights')
                self.bias12 = bias_layer([512], 'bias')
                self.conv12 = conv_layer(self.conv11, self.weights12, self.bias12)
            self.pool4 = max_pool(self.conv12, 'pool4')

        with tf.variable_scope("group5"):
            self.fc6 = tf.reshape(self.pool4, [-1, middle_shape], 'fc6')
            with tf.variable_scope("fc7"):
                self.weights13 = kernel_layer([middle_shape, 4096], 'weights')
                self.bias13 = bias_layer(4096, 'bias')
                self.fc7 = matmul(self.fc6, self.weights13, self.bias13)
            with tf.variable_scope("logits"):
                self.weights14 = kernel_layer([4096, n_classes], 'weights')
                self.bias14 = bias_layer([n_classes], 'bias')
                self.logits = matmul(self.fc7, self.weights14, self.bias14)
        self.softmax = tf.nn.softmax(self.logits, 'softmax')


if __name__ == "__main__":
    vgg = VGG(10)
