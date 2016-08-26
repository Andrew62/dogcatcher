#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""

import os
import time
import util
import shutil
import traceback
import tensorflow as tf
from data import DataSet
from msg import send_mail
from encoder import OneHot
from datetime import datetime


def train_model(class_pkl, train_pkl, model, model_dir, debug=False):
    if debug is True:
        print "DEBUG MODE"
        MESSAGE_EVERY = 1
        EMAILING = False
        BATCH_SIZE = 16
        SAVE_ITER = 30
        EPOCHS = 1
    else:
        MESSAGE_EVERY = 25
        EMAILING = True
        BATCH_SIZE = 16
        SAVE_ITER = 5000
        EPOCHS = 90

    EMAIL_EVERY = MESSAGE_EVERY * 250
    n_classes = 252
    train = True

    classes = util.pkl_load(class_pkl)
    encoder = OneHot(classes)
    train_set = util.pkl_load(train_pkl)
    data = DataSet(train_set, BATCH_SIZE, EPOCHS)

    graph = tf.Graph()
    with graph.as_default():
        model = model(n_classes, train=train)
        train_labels_placeholder = tf.placeholder(dtype=tf.float32, name="train_labels", shape=[BATCH_SIZE, n_classes])
        learning_rate_placeholder = tf.placeholder(dtype=tf.float32, name='learning_rate')

        tf.scalar_summary('learning_rate', learning_rate_placeholder)
        tf.image_summary("raw_input", model.input_data)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model.logits, train_labels_placeholder))
        tf.scalar_summary('loss', loss)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        trainable_vars = tf.trainable_variables()

        saver = tf.train.Saver(trainable_vars)
        initop = tf.initialize_all_variables()
        merged = tf.merge_all_summaries()

    sess = tf.Session(graph=graph)
    summary_writer = tf.train.SummaryWriter(os.path.join(model_dir, 'summary', time.strftime("%Y%m%d%H%M%S")),
                                            graph=graph)
    with sess.as_default():
        sess.run(initop)
        print "\n" + "*" * 50
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt is not None:
            print "\nCheckpoint {0} restored!".format(os.path.basename(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Initialized"

        print "Training on {0} exmaples".format(len(data.data))

        print "\n", "*" * 50
        print "Batch size: {0} images".format(BATCH_SIZE)
        epoch = 0
        i = 0
        data.start()
        learning_rate = 0.01
        try:
            while epoch <= EPOCHS:
                start = time.time()
                train_data, train_labels, epoch = data.batch()
                train_lab_vec = encoder.encode(train_labels)
                feed = {model.input_data: train_data,
                        train_labels_placeholder: train_lab_vec,
                        learning_rate_placeholder: learning_rate}
                _, sess_loss, predictions, summary = sess.run([optimizer, loss, model.softmax, merged],
                                                              feed_dict=feed)
                summary_writer.add_summary(summary, i)

                if ((i + 1) % MESSAGE_EVERY == 0) or (i == 0):
                    avg_loss = sess_loss.mean()
                    total_correct, minibatch_accuracy = util.accuracy(predictions, train_lab_vec)
                    subj, msg = util.get_message(i, minibatch_accuracy, start, avg_loss,
                                                 total_correct, epoch, learning_rate)
                    print msg
                    if (((i + 1) % EMAIL_EVERY) == 0) and (EMAILING is True):
                        send_mail("dogcatcher update: " + subj, msg)

                if ((i + 1) % SAVE_ITER) == 0:
                    saver.save(sess, os.path.join(model_dir, util.model_name(datetime.now())))
                    print "\n" + "*" * 50
                    print "Successful checkpoint iteration {0}".format(i + 1)

                if (epoch + 1) % 15 == 0:
                    learning_rate *= 0.1
                i += 1

            msg = "\n" + "*" * 50
            msg += "\n" + "*" * 50
            subj = "Training complete!"
            print msg
        except Exception as e:
            traceback.print_exc()
            subj = "DOGCATCHER STOPPED!"
            msg = "Failed after {0} steps".format(i)
            print msg

        finally:
            saver.save(sess, os.path.join(model_dir, util.model_name(datetime.now())))
            outg = os.path.join(model_dir, "graph")
            if os.path.exists(outg):
                shutil.rmtree(outg)
                tf.train.write_graph(sess.graph_def, outg, "graph.pb")
            util.pkl_dump(encoder, os.path.join(model_dir, "encoder.pkl"))
            if EMAILING is True:
                send_mail(subj, msg)
