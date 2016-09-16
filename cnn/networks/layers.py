#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""

import tensorflow as tf
from operator import mul


def conv2d(inputs, n_out, kernel_height, kernel_width, **kwargs):
    """
    :param inputs: input layer
    :param kernel_height: conv filter kernel width
    :param kernel_width: conv filter kernel height
    :param channles_out: conv depth
    :param name: layer name
    :param conv_height: verticle steps for conv filter
    :param conv_width: horizontal steps for conv filter
    :param relu: whether or not to apply Relu
    :param bias: whether or not to apply bias
    :param padding: padding to use for conv. Either "SAME" or "VALID"
    :return: conv layer
    """
    name = kwargs.pop('name','conv')
    conv_height = kwargs.pop('conv_height', 1)
    conv_width = kwargs.pop('conv_width', 1)
    relu = kwargs.pop('relu', True)
    bias = kwargs.pop('bias', True)
    padding = kwargs.pop("padding", "SAME")
    scope = kwargs.pop('scope', None)
    reuse = kwargs.pop('reuse', None)

    channles_in = inputs.get_shape().as_list()[-1]

    with tf.variable_op_scope([inputs], scope, name, reuse=reuse) as scope:
        weights = tf.Variable(tf.truncated_normal([kernel_height, kernel_width, channles_in, n_out],
                                                  stddev=0.01), name='weights')
        convolve = tf.nn.conv2d(inputs, weights, [1, conv_height, conv_width, 1],
                                padding=padding)
        if bias is True:
            bias_layer = tf.Variable(tf.constant(0.001, dtype=tf.float32, shape=[n_out]), name='biases')
            convolve = tf.nn.bias_add(convolve, bias_layer)

        if relu is True:
            convolve = tf.nn.relu(convolve)

    return convolve


def affine(inputs, n_out, **kwargs):
    """

    :param inputs: tensor
    :param n_out: shape of outputs
    :param name: layer name
    :param bias: bool. Use bias or not
    :param relu: bool. Use relu or not
    :return: affine tensor
    """
    name = kwargs.pop("name", "affine")
    bias = kwargs.pop("bias", True)
    relu = kwargs.pop("relu", True)
    scope = kwargs.pop('scope', None)
    reuse = kwargs.pop('reuse', None)

    input_shape = inputs.get_shape().as_list()
    if len(input_shape) == 4:
        n_in = reduce(mul, input_shape[1:], 1)
        inputs = tf.reshape(inputs, shape=[-1, n_in])
    else:
        n_in = input_shape[-1]

    with tf.variable_op_scope([inputs], scope, name, reuse=reuse) as scope:
        weights = tf.Variable(tf.truncated_normal([n_in, n_out],
                                                  stddev=0.01), name='weights')
        fc = tf.matmul(inputs, weights)

        if bias is True:
            bias_layer = tf.Variable(tf.constant(0.001, dtype=tf.float32, shape=[n_out]), name='biases')
            fc = tf.nn.bias_add(fc, bias_layer)

        if relu is True:
            fc = tf.nn.relu(fc)

    return fc


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)
