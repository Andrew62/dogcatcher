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

n_epochs = 60
batch_size = 16
message_every = 50

pretrained_weights_npy = workspace.vgg16c_weights
model_dir = workspace.vgg16c_models
classes = util.pkl_load(workspace.class_pkl)
data = DataSet(workspace.train_pkl, batch_size, n_epochs)

encoder = OneHot(classes)


graph = tf.Graph()
with graph.as_default():
    model = VGG16_D(len(classes))
    train_labels_placeholder = tf.placeholder(dtype=tf.float32, name="train_labels")
    learning_rate_placeholder = tf.placeholder(dtype=tf.float32, name='learning_rate')

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model.logits, train_labels_placeholder))
    tf.scalar_summary('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder).minimize(loss)
    variables = tf.all_variables()
    saver = tf.train.Saver(variables, keep_checkpoint_every_n_hours=24)
    initop = tf.initialize_all_variables()
    merged = tf.merge_all_summaries()

sess = tf.Session(graph=graph)
summary_writer = tf.train.SummaryWriter(os.path.join(model_dir, 'summary', time.strftime("%Y%m%d%H%M%S")),
                                        graph=graph)
with sess.as_default():
    try:
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
                if 'fc7' in op.name:
                    break
                print op.name
                layer_split = op.name.split("/")
                layer, var = layer_split[0], layer_split[1].split(":")[0]
                sess.run(op.assign(pretrained_features[layer][var]))

        i = 0
        epochs = 0
        data.start()
        learning_rate = 0.01
        while epochs <= n_epochs:
            start = time.time()
            train_data, train_labels, epoch = data.batch()
            train_lab_vec = encoder.encode(train_labels)
            feed = {model.input_data: train_data,
                    train_labels_placeholder: train_lab_vec,
                    learning_rate_placeholder: learning_rate}
            _, sess_loss, predictions, summary = sess.run([optimizer, loss, model.softmax, merged],
                                                          feed_dict=feed)
            summary_writer.add_summary(summary, i)
            if ((i + 1) % message_every == 0) or (i == 0):
                avg_loss = sess_loss.mean()
                total_correct, minibatch_accuracy = util.accuracy(predictions, train_lab_vec)
                subj, msg = util.get_message(i, minibatch_accuracy, start, avg_loss, total_correct)
                print msg

            if (epoch + 1) % 30 == 0:
                learning_rate *= 0.1
            i += 1

    except Exception as e:
        raise e

    finally:
        print("Cleaning up...")
        saver.save(sess, os.path.join(model_dir, "vgg16.model"))
        # tf.train.write_graph(sess.graph_def, model_dir, "vgg16.graph", as_text=False)
        print("Complete!")
