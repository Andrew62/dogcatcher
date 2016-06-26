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
from cnn.tf_data import train_image_producer

data = pkl_load(workspace.test_pkl)
encoder = OneHot(pkl_load(workspace.class_pkl))

labels = data[:100, 0]
paths = data[:100, 1]

# data = tf.train.slice_input_producer([tf.constant(labels), tf.constant(paths)], num_epochs=1)
# batch = tf.train.batch(data, batch_size=32)
imgs, labs = train_image_producer(paths, labels, batch_size=32, epochs=1)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    sess.run(init)

    start = time.time()
    i = 0.0
    while True:
        try:
            print labs
            print imgs
            i += 1
        except tf.errors.OutOfRangeError:
            break
    print "Average load time: {0:0.2}".format((time.time()-start)/i)
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=2)
