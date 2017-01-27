#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""

import pickle
from cnn import datatf
import tensorflow as tf

# load class pickle index. Used to map the
# indicies of the training data from one-hot index
# values to labels
with open("test_data/dogs/classes.pkl", 'rb') as infile:
    class_idx = pickle.load(infile)

data_path = "test_data/dogs/dogs.csv"

# simple graph to test the loading op
graph = tf.Graph()
with graph.as_default():
    # load batches of labels and images
    one_hot_targets, image_batch = datatf.batch_producer([data_path], len(class_idx))

    # get the max for each onehot row so we can convert back to text
    idxs = tf.argmax(one_hot_targets, 1)


with tf.Session(graph=graph) as sess:
    # required to run the data loader
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    # run in an infinite loop until we've exceeded the number of epochs
    # When the epoch limit is reached, function throws and OutOfRangeError
    while True:
        try:
            idx_np = sess.run(idxs)
        except tf.errors.OutOfRangeError:
            break

    # clean up the data loader
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=2)

    # print to make sure everything is running properly
    print("\n".join(map(lambda x: class_idx[x], list(idx_np))))

