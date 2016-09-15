"""
A tensorflow implementation of VGG19. Structure was pulled
from the paper.
"""

import layers
import tensorflow as tf


class VGG16_D(object):
    def __init__(self, n_classes=1000, train=False):

        self.layers = {}

        self.input_data = tf.placeholder(dtype=tf.float32, name='input_data', shape=[None, 224, 224, 3])

        batch_norm = self.apply_batch_norm(self.input_data)
        self.layers['group1'] = self.pooling_group(batch_norm, 64, 'group1')
        self.layers['group2'] = self.pooling_group(self.layers['group1'], 128, 'group2')
        self.layers['group3'] = self.pooling_group(self.layers['group2'], 128, 'group3', 3)
        self.layers['group4'] = self.pooling_group(self.layers['group3'], 256, 'group4', 3)
        self.layers['group5'] = self.pooling_group(self.layers['group4'], 512, 'group5', 3)
        self.layers['group6'] = self.pooling_group(self.layers['group5'], 512, 'group6', 3)
        self.layers['fc1'] = layers.affine(self.layers['group6'], 4096, name='fc1')
        self.layers['fc2'] = layers.affine(self.layers['fc1'], 4096, name='fc1')
        self.logits = layers.affine(self.layers['fc2'], n_classes, name='logits', relu=False)
        self.softmax = tf.nn.softmax(self.logits, name='softmax')

    def pooling_group(self, inputs, n_out, name, n_conv=2):
        with tf.variable_op_scope([inputs], name, 'PoolingGroup') as sc:
            layer_collection = {}
            current_input = inputs
            for i in range(n_conv):
                layer_collection['conv{0}'.format(i + 1)] = layers.conv2d(current_input, n_out, 3, 3, scope=sc)
                current_input = layer_collection['conv{0}'.format(i + 1)]
            layer_collection['pool'] = tf.nn.max_pool(current_input, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
        layers.variable_summaries(layer_collection['pool'], name)
        self.layers[name] = layer_collection
        return layer_collection['pool']

    def apply_batch_norm(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[0, 1, 2])
        batch_norm = tf.nn.batch_normalization(inputs, mean, var, offset=1, scale=1,
                                               variance_epsilon=1e-6)
        return batch_norm




if __name__ == "__main__":
    vgg = VGG16_D(252)

    print vgg.mean_subtract.get_shape()

    for l in vgg.layers:
        print l.name, l.get_shape()
