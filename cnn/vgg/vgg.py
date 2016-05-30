
"""
A tensorflow implementation of VGG19. Structure was pulled
from the paper.
"""

import tensorflow as tf
from cnn.vgg.wrapper import kernel, bias, conv_layer, max_pool, matmul


class VGG(object):
    def __init__(self, n_classes, middle_shape=4096):

        self.layers = {
            1 : kernel([3, 3, 3, 64], 'conv1'),
            2 : kernel([3, 3, 64, 64], 'conv2'),

            3 : kernel([3, 3, 64, 128], 'conv3'),
            4 : kernel([3, 3, 128, 128], 'conv4'),

            5 : kernel([3, 3, 128, 256], 'conv5'),
            6 : kernel([3, 3, 256, 256], 'conv6'),
            7 : kernel([3, 3, 256, 256], 'conv7'),
            8 : kernel([3, 3, 256, 256], 'conv8'),

            9 : kernel([3, 3, 256, 512], 'conv9'),
            10 : kernel([3, 3, 512, 512], 'conv10'),
            11 : kernel([3, 3, 512, 512], 'conv11'),
            12 : kernel([3, 3, 512, 512], 'conv12'),

            #fc
            13 : kernel([middle_shape, 4096], 'fc7'),
            14 : kernel([4096, n_classes], 'fc8')

        }

        self.biases = {
            1 : bias([64], 'bias1'),
            2 : bias([64], 'bias2'),

            3 : bias([128], 'bias3'),
            4 : bias([128], 'bias4'),

            5 : bias([256], 'bias5'),
            6 : bias([256], 'bias6'),
            7 : bias([256], 'bias7'),
            8 : bias([256], 'bias8'),

            9 : bias([512], 'bias9'),
            10 : bias([512], 'bias10'),
            11 : bias([512], 'bias11'),
            12 : bias([512], 'bias12'),

            13 : bias([4096], 'bias13'),
            14 : bias([n_classes], 'bias14')

        }

    def predict(self, data):

        # Don't need to specify input data shape once we know everything is hooked up
        # For reference, the verification shape is [256, 224, 224, 3]

        # TODO move these out of the method
        # self.train_labels_placeholder = tf.placeholder(dtype=tf.float32, name="train_labels_placeholder", shape=[10, 224, 224, 3])
        # self.train_data_placeholder = tf.placeholder(dtype=tf.float32, name="train_data_placeholder", shape=[10, 224, 224, 3])

        # TODO add validation and test inputs

        conv1 = conv_layer(data, self.layers[1], self.biases[1], '1')
        conv2 = conv_layer(conv1, self.layers[2], self.biases[2], '2')
        pool1 = max_pool(conv2, 'pool1')

        conv3 = conv_layer(pool1, self.layers[3], self.biases[3], '3')
        conv4 = conv_layer(conv3, self.layers[4], self.biases[4], '4')
        pool2 = max_pool(conv4, 'pool2')

        conv5 = conv_layer(pool2, self.layers[5], self.biases[5], '5')
        conv6 = conv_layer(conv5, self.layers[6], self.biases[6], '6')
        conv7 = conv_layer(conv6, self.layers[7], self.biases[7], '7')
        conv8 = conv_layer(conv7, self.layers[8], self.biases[8], '8')
        pool3 = max_pool(conv8, 'pool3')

        conv9 = conv_layer(pool3, self.layers[9], self.biases[9], '9')
        conv10 = conv_layer(conv9, self.layers[10], self.biases[10], '10')
        conv11 = conv_layer(conv10, self.layers[11], self.biases[11], '11')
        conv12 = conv_layer(conv11, self.layers[12], self.biases[12], '12')
        pool3 = max_pool(conv12, 'pool4')

        fc6 = tf.reshape(pool3, [1, -1], 'fc6')
        fc7 = matmul(fc6, self.layers[13], self.biases[13], 'fc8')
        return matmul(fc7, self.layers[14], self.biases[14], 'logtis')

        # TODO remove the code below and put in training script

        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.train_labels_placeholder))

        # optimizer = tf.train.AdagradOptimizer(learning_rate=1e-3).minimize(self.loss)

        # prediction = tf.nn.softmax(self.logits)



if __name__ == "__main__":
    vgg = VGG(10)
