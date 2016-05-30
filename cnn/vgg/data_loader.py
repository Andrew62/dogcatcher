#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""

import util
import tensorflow as tf


def input_pipline(fp, encoder, batch_size=256):
    """
    Reads input csv and returns an input queue
    """
    paths = []
    labels = []
    for label, path in util.pkl_load(fp):
        paths.append(path)
        labels.append(label)
    labels = encoder.encode(labels)
    paths_tf = tf.convert_to_tensor(paths, dtype=tf.string)
    labels_tf = tf.convert_to_tensor(labels, dtype=tf.float32)
    producer = tf.train.slice_input_producer([paths_tf, labels_tf], shuffle=True)
    image, label = img_prep(producer)
    image_batch, label_batch = tf.train.batch([image, label], batch_size, capacity=(3*batch_size))

    return image_batch, label_batch


def img_prep(queue):
    label = queue[1]
    path = queue[0]
    file_contents = tf.read_file(path)
    img = tf.image.decode_jpeg(file_contents, channels=3)
    img_converted = tf.image.convert_image_dtype(img,  dtype=tf.float32)
    mean = tf.reduce_mean(img_converted)
    return_img = img_converted - mean
    return_img.set_shape([256, 256, 3])
    return return_img, label

