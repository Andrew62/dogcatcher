"""
Tensorflow wrapper implemented independent of the alexnet wrapper.
Any VGG specific support code will go in the VGG folder
"""

import numpy as np
import pickle as pkl
import tensorflow as tf

def load_vgg_layers(path):
    """
    Loads VGG19 layers except the last
    :param path: path to vgg19 pickl
    :return: list of matrices
    """
    with open(path, 'rb') as infile:
        vgg = pkl.load(infile)

    return vgg['param values'][:-2]

def load_variable(var, name):
    if len(var.shape) < 2:
        shape = [var.shape[0], 1]
        var = var.reshape(shape)
    return tf.Variable(var, name=name, dtype=tf.float32)

def kernel(shape, name):
    """
    Initializes a variable
    :param shape: [kernel width, kernel height, kernel depth, output depth]
    :param name: var name with no spaces
    :return: tensorflow variable
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=1e-2,  name=name, dtype=tf.float32))

def bias(shape, name):
    return tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=shape, name=name))

def conv_layer(input, filter, bias, name, strides=[1,1,1,1]):
    with tf.variable_scope(name) as scope:
        conv = tf.nn.conv2d(input, filter, strides, padding='SAME')
        return tf.nn.elu(conv + bias)

def matmul(a, b, bias, name):
    with tf.variable_scope(name) as scope:
        mult = tf.matmul(a, b)
        return tf.nn.elu(mult + bias)

def max_pool(input, name):
    return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1],
                          padding='SAME', name=name)
def accuracy(predictions, labels):
    return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) * 1.) / predictions.shape[0]


def pkl_dump(obj, fp):
    with open(fp, 'wb') as target:
        pkl.dump(obj, target)

def pkl_load(fp):
    with open(fp, 'rb') as infile:
        return pkl.load(infile)

def placeholder(name):
    return tf.placeholder(dtype=tf.float32, name=name)

def constant(value, name, shape=None):
    return tf.constant(value, name=name, dtype=tf.float32, shape=shape)

def __make_cls_pkl__():
    import csv
    from config import workspace
    classes = set()
    with open(workspace.data_csv, 'rb') as infile:
        reader = csv.reader(infile)
        for path, cls in reader:
            classes.add(cls)
    classes = list(classes)
    pkl_dump(classes, workspace.class_pkl)

if __name__ == "__main__":
    __make_cls_pkl__()