"""
A tensorflow implementation of VGG19. Structure was pulled
from the paper.
"""

import layers
import tensorflow as tf


class VGG16_D(object):
    def __init__(self, n_classes=1000, train=False):

        self.input_data = tf.placeholder(dtype=tf.float32, name='input_data', shape=[None, 224, 224, 3])

        mean = tf.constant([125.974950491, 121.990847064, 102.991749558],
                           dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')

        self.mean_subtract = self.input_data - mean

        self.layers = []

        inputs = self.input_data
        for i, n_out in enumerate([64, 128, 256, 512, 512]):
            conv = layers.conv2d(inputs, n_out, 3, 3, name="conv_{}_1".format(i))
            self.layers.append(conv)
            conv = layers.conv2d(conv, n_out, 3, 3, name="conv_{}_2".format(i))
            self.layers.append(conv)

            if n_out > 128:
                conv = layers.conv2d(conv, n_out, 3, 3, name="conv_{}_3".format(i))
                self.layers.append(conv)

            inputs = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
            self.layers.append(inputs)

        for i in range(2):
            inputs = layers.affine(inputs, 4096, name="fc_{}".format(i + 1))
            if train is True:
                inputs = tf.nn.dropout(inputs, keep_prob=0.5)
            self.layers.append(inputs)

        self.logits = layers.affine(inputs, n_classes, relu=False)

        self.softmax = tf.nn.softmax(self.logits, 'softmax')


if __name__ == "__main__":
    vgg = VGG16_D(252)

    for l in vgg.layers:
        print l.name
