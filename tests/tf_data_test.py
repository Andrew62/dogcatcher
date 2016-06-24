#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""
from __future__ import absolute_import

import time
import tensorflow as tf
from cnn.util import pkl_load
from cnn.encoder import OneHot
from cnn.config import  workspace
from cnn.tf_data import ImageProducer

data = pkl_load(workspace.test_pkl)
encoder = OneHot(pkl_load(workspace.class_pkl))

labels = data[:, 0]
paths = data[:, 1]

producer = ImageProducer(paths, labels, batch_size=64, n_processes=4)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = producer.start(sess, coord)

    iters = 0
    epochs = 0
    start = time.time()
    while epochs < 1:
        try:
            labels, images = producer.get_batch(sess)
            lab_encode = encoder.encode(labels)
            print lab_encode.shape, images.shape
            iters += 1
        except tf.errors.OutOfRangeError:
            epochs += 1
            print "Epoch {0} complete".format(epochs)
    print "Average load time: {0:0.2f} secs".format((time.time()-start)/iters)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=2)
