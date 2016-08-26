"""
A tensorflow implementation of VGG19. Structure was pulled
from the paper.
"""

import layers
import tensorflow as tf


class VGG16_D(object):
    def __init__(self, n_classes=1000, train=False):

        self.input_data = tf.placeholder(dtype=tf.float32, name='input_data', shape=[None, 224, 224, 3])

        name = 'batch_norm'
        self.layer_names.append(name)
        with tf.variable_scope(name):
            mean, var = tf.nn.moments(self.input_data, axes=[0, 1, 2])
            self.batch_norm = tf.nn.batch_normalization(self.input_data, mean, var, offset=1, scale=1,
                                                        variance_epsilon=1e-6)
        tf.image_summary(name, self.batch_norm)

        if train is True:
            tf.image_summary("mean_subtract", self.batch_norm)

        self.layers = []

        inputs = self.batch_norm
        for i, n_out in enumerate([64, 128, 256, 512, 512]):
            with tf.variable_scope("pool_{}".format(i + 1)):
                conv = layers.conv2d(inputs, n_out, 3, 3, name="conv_{}_1".format(i))
                self.layers.append(conv)
                conv = layers.conv2d(conv, n_out, 3, 3, name="conv_{}_2".format(i))
                self.layers.append(conv)

                if n_out > 128:
                    conv = layers.conv2d(conv, n_out, 3, 3, name="conv_{}_3".format(i))
                    self.layers.append(conv)

                inputs = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
                self.layers.append(inputs)

                if train is True:
                    layers.variable_summaries(inputs, "pool_{}".format(i + 1))

        for i in range(2):
            inputs = layers.affine(inputs, 4096, name="fc_{}".format(i + 1))
            if train is True:
                inputs = tf.nn.dropout(inputs, keep_prob=0.5)
            self.layers.append(inputs)

        self.logits = layers.affine(inputs, n_classes, relu=False)

        self.softmax = tf.nn.softmax(self.logits, 'softmax')


if __name__ == "__main__":
    vgg = VGG16_D(252)

    print vgg.mean_subtract.get_shape()

    for l in vgg.layers:
        print l.name, l.get_shape()
