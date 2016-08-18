#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""

import os
import time
import util
import numpy as np
import tensorflow as tf
from data import DataSet
from encoder import OneHot
from config import workspace
from networks.vgg16_finetune import VGG16_D

n_epochs = 1
batch_size = 8
message_every = 1

pretrained_weights_npy = workspace.vgg16c_weights
model_dir = workspace.vgg16c_models
classes = util.pkl_load(workspace.class_pkl)
data = DataSet(workspace.train_pkl, batch_size, n_epochs)

encoder = OneHot(classes)

graph = tf.Graph()
with graph.as_default():
    model = VGG16_D(1000)
    # train_labels_placeholder = tf.placeholder(dtype=tf.float32, name="train_labels")
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model.logits, train_labels_placeholder))
    # tf.scalar_summary('loss', loss)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    variables = tf.all_variables()
    saver = tf.train.Saver(variables, keep_checkpoint_every_n_hours=24)
    initop = tf.initialize_all_variables()
    merged = tf.merge_all_summaries()

sess = tf.Session(graph=graph)
summary_writer = tf.train.SummaryWriter(os.path.join(model_dir, 'summary', time.strftime("%Y%m%d%H%M%S")),
                                        graph=graph)
with sess.as_default():
    sess.run([initop])
    checkpoint = tf.train.get_checkpoint_state(model_dir)

    if checkpoint is not None:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Weights restored!")
    else:
        print "Loading pretrained..."
        pretrained_features = np.load(pretrained_weights_npy).item()
        # This will map the numpy VGG16 weights to the tensorflow ops
        # This mapping is weak and should use something more robust
        # like a tensorflow graph or checkpoint object
        for op in variables:
            # layers are loaded sequentially
            if 'beta' in op.name:
                break
            print op.name
            layer_split = op.name.split("/")
            layer, var = layer_split[0], layer_split[1].split(":")[0]
            sess.run(op.assign(pretrained_features[layer][var]))

    i = 0
    epochs = 0
    data.start()
    out_array = None
    while epochs < n_epochs:
        start = time.time()
        train_data, train_labels, epoch = data.batch()
        train_lab_vec = encoder.encode(train_labels)
        feed = {model.input_data: train_data}
        features, summary = sess.run([model.logits, merged], feed_dict=feed)
        if i == 0:
            out_array = features
        else:
            out_array = np.concatenate((out_array, features))

        print "{0:0.2%} complete ({1:0.1f} seconds)".format((batch_size * (i + 1.0)) / len(data.data),
                                                            time.time() - start)
        i += 1

        if (i + 1) == 100:
            break

    np.save("../models/vgg_finetune/vectors.npy", out_array)
    saver.save(sess, os.path.join(model_dir, "vgg16.model"))
    print("Complete!")
