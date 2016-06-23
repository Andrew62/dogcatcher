#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""

import os
import util
import time
import shutil
import helper
import tensorflow as tf
from data import DataSet
from msg import send_mail
from encoder import OneHot
from alexnet import AlxNet
from config import workspace
from datetime import datetime


def get_message(i, minibatch_accuracy, start, avg_loss):
    subj = 'Iteration {0} Minibatch accuracy: {1:0.2%}'.format(i + 1, minibatch_accuracy)
    msg = "\n" + "*" * 50
    msg += '\nMinibatch loss at step {0}: {1:0.9f}\n'.format(i + 1, avg_loss)
    msg += subj + '\n'
    msg += 'Minibatch time: {0:0.0f} secs\n'.format(time.time() - start)
    msg += time.ctime()
    return subj, msg


def train_alexnet(debug=False):
    if debug is True:
        print "DEBUG MODE"
        MESSAGE_EVERY = 1
        EMAILING = False
        BATCH_SIZE = 32
        SAVE_ITER = 30
        EPOCHS = 1
    else:
        MESSAGE_EVERY = 25
        EMAILING = True
        BATCH_SIZE = 128
        SAVE_ITER = 1000
        EPOCHS = 90

    EMAIL_EVERY = MESSAGE_EVERY * 80
    n_classes = 252
    train = True

    classes = util.pkl_load(workspace.class_pkl)
    encoder = OneHot(classes)
    data = DataSet(workspace.train_pkl, workspace.test_pkl,
                   workspace.valid_pkl, workspace.class_pkl,
                   img_shape=(224, 224, 3))

    graph = tf.Graph()
    with graph.as_default():
        model = AlxNet(n_classes, train=train)
        train_labels_placeholder = tf.placeholder(dtype=tf.float32, name="train_labels")

        tf.image_summary("raw_input", model.input_data)
        tf.image_summary("normed_data", model.batch_norm)

        model_layers = ['response_norm1', 'pool1', 'response_norm2', 'pool2',
                         'hidden3', 'hidden4', 'hidden5', 'pool5', 'fc6', 'fc7']
        
        for layer in model_layers:
            helper.var_summary(getattr(model, layer), layer)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model.logits, train_labels_placeholder))
        tf.scalar_summary('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

        saver = tf.train.Saver()
        initop = tf.initialize_all_variables()
        merged = tf.merge_all_summaries()

    sess = tf.Session(graph=graph)
    summary_writer = tf.train.SummaryWriter(os.path.join(workspace.alexnet_models, "summaries",
                                                         time.strftime("%Y%m%d%H%M%S")), graph=graph)
    with sess.as_default():
        sess.run(initop)
        print "\n" + "*" * 50
        ckpt = tf.train.get_checkpoint_state(workspace.alexnet_models)
        if ckpt is not None:
            print "\nCheckpoint {0} restored!".format(os.path.basename(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Initialized"

        print "Training on {0} exmaples".format(len(data.tracker['train']['data']))

        print "\n", "*" * 50
        print "Batch size: {0} images".format(BATCH_SIZE)
        epoch = 0
        i = 0
        try:
            while epoch <= EPOCHS:
                start = time.time()
                train_data, train_labels, epoch = data.train_batch(BATCH_SIZE)
                train_lab_vec = encoder.encode(train_labels)

                feed = {model.input_data: train_data,
                        train_labels_placeholder: train_lab_vec}
                _, sess_loss, predictions, summary = sess.run([optimizer, loss, model.softmax, merged],
                                                              feed_dict=feed)

                if ((i + 1) % MESSAGE_EVERY == 0) or (i == 0):
                    avg_loss = sess_loss.mean()
                    minibatch_accuracy = util.accuracy(predictions, train_lab_vec)
                    subj, msg = get_message(i, minibatch_accuracy, start, avg_loss)
                    print msg
                    summary_writer.add_summary(summary, i)

                    if (((i + 1) % EMAIL_EVERY) == 0) and (EMAILING is True):
                        send_mail("dogcatcher update: " + subj, msg)

                if ((i + 1) % SAVE_ITER) == 0:
                    saver.save(sess, os.path.join(workspace.alexnet_models, util.model_name(datetime.now())))
                    print "\n" + "*" * 50
                    print "Successful checkpoint iteration {0}".format(i + 1)
                i += 1
            msg = "\n" + "*" * 50
            msg += "\n" + "*" * 50
            # msg += "\nTest accuracy: {0:0.2%}".format(util.accuracy(test_prediction.eval(), test_lab_vec))
            subj = "Training complete!"
            print msg
        except Exception as e:
            print e
            subj = "DOGCATCHER STOPPED!"
            msg = "Failed after {0} steps".format(i)
            print msg

        finally:
            saver.save(sess, os.path.join(workspace.alexnet_models, util.model_name(datetime.now())))
            outg = os.path.join(workspace.alexnet_models, "graph")
            if os.path.exists(outg):
                shutil.rmtree(outg)
                tf.train.write_graph(sess.graph_def, outg, "graph.pb")
            util.pkl_dump(encoder, os.path.join(workspace.alexnet_models, "encoder.pkl"))
            if EMAILING is True:
                send_mail(subj, msg)