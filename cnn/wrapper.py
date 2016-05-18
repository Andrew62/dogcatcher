"""
This file wraps Tensorflow objects for better readability
"""

import tensorflow as tf


def kernel(shape, name):
    if type(shape) is not list:
        raise TypeError("shape must be a list not {0}".format(type(shape)))
    return tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=1e-2, name=name))

def bias(shape, name, constant=0):
    if type(shape) is not list:
        raise TypeError("shape must be a list not {0}".format(type(shape)))
    return tf.Variable(tf.constant(constant, shape=shape, dtype=tf.float32, name=name))

def norm(input, name):
    return tf.nn.local_response_normalization(input, depth_radius = 2,alpha = 2e-05,
                                              beta = 0.75, bias = 1.0, name=name)

def max_pool(input, name):
    return tf.nn.max_pool(input, [1, 3, 3, 1], [1, 2, 2, 1], "SAME", name=name)

def placeholder(name):
    return tf.placeholder(dtype=tf.float32, name=name)

def constant(value, name, shape=None):
    return tf.constant(value, name=name, dtype=tf.float32, shape=shape)