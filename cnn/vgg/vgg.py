
"""
A tensorflow implementation of VGG19. Structure was pulled
from the paper.
"""

import tensorflow as tf
from .wrapper import conv_layer, max_pool, matmul


class VGG(object):
    def __init__(self, n_classes, middle_shape=4096):

        self.middle_shape = middle_shape

        self.shapes = {
            1 : [3, 3, 3, 64],
            2 : [3, 3, 64, 64],

            3 : [3, 3, 64, 128],
            4 : [3, 3, 128, 128],

            5 : [3, 3, 128, 256],
            6 : [3, 3, 256, 256],
            7 : [3, 3, 256, 256],
            8 : [3, 3, 256, 256],

            9 : [3, 3, 256, 512],
            10 : [3, 3, 512, 512],
            11 : [3, 3, 512, 512],
            12 : [3, 3, 512, 512],

            #fc
            13 : [middle_shape, 4096],
            14 : [4096, n_classes],

        }

    def predict(self, data):

        # Don't need to specify input data shape once we know everything is hooked up
        # For reference, the verification shape is [256, 224, 224, 3]

        # TODO move these out of the method
        # self.train_labels_placeholder = tf.placeholder(dtype=tf.float32, name="train_labels_placeholder", shape=[10, 224, 224, 3])
        # self.train_data_placeholder = tf.placeholder(dtype=tf.float32, name="train_data_placeholder", shape=[10, 224, 224, 3])

        # TODO add validation and test inputs

        # TODO move scope from wrapper out to here and wrap each of these. move kernel and bias creetion into
        # TODO the below conv_layer, max_pool, etc

        with tf.variable_scope("group1"):
            with tf.variable_scope("conv1"):
                conv1 = conv_layer(data, self.shapes[1])
            with tf.variable_scope("conv2"):
                conv2 = conv_layer(conv1, self.shapes[2])
            pool1 = max_pool(conv2, 'pool1')

        with tf.variable_scope("group2"):
            with tf.variable_scope("conv3"):
                conv3 = conv_layer(pool1, self.shapes[3])
            with tf.variable_scope("conv4"):
                conv4 = conv_layer(conv3, self.shapes[4])
            pool2 = max_pool(conv4, 'pool2')

        with tf.variable_scope("group3"):
            with tf.variable_scope("conv5"):
                conv5 = conv_layer(pool2, self.shapes[5])
            with tf.variable_scope("conv6"):
                conv6 = conv_layer(conv5, self.shapes[6])
            with tf.variable_scope("conv7"):
                conv7 = conv_layer(conv6, self.shapes[7])
            with tf.variable_scope("conv8"):
                conv8 = conv_layer(conv7, self.shapes[8])
            pool3 = max_pool(conv8, 'pool3')

        with tf.variable_scope("group4"):
            with tf.variable_scope("conv9"):
                conv9 = conv_layer(pool3, self.shapes[9])
            with tf.variable_scope("conv10"):
                conv10 = conv_layer(conv9, self.shapes[10])
            with tf.variable_scope("conv11"):
                conv11 = conv_layer(conv10, self.shapes[11])
            with tf.variable_scope("conv12"):
                conv12 = conv_layer(conv11, self.shapes[12])
            pool4 = max_pool(conv12, 'pool4')

        with tf.variable_scope("group5"):
            fc6 = tf.reshape(pool4, [-1, self.middle_shape], 'fc6')
            with tf.variable_scope("fc7"):
                fc7 = matmul(fc6, self.shapes[13])
            with tf.variable_scope("logits"):
                return matmul(fc7, self.shapes[14])

        # TODO remove the code below and put in training script

        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.train_labels_placeholder))

        # optimizer = tf.train.AdagradOptimizer(learning_rate=1e-3).minimize(self.loss)

        # prediction = tf.nn.softmax(self.logits)



if __name__ == "__main__":
    vgg = VGG(10)
