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
import tensorflow as tf
from data import DataSet
from msg import send_mail
from encoder import OneHot
from .alexnet import AlxNet
from config import workspace
from datetime import datetime


def train_alexnet(debug=False):
    if debug is True:
        print "DEBUG MODE"
        MESSAGE_EVERY = 1
        EMAILING = False
        TRAIN_BATCH_SIZE = 32
        SAVE_ITER = 30
        EPOCHS = 1
    else:
        MESSAGE_EVERY = 50
        EMAILING = True
        TRAIN_BATCH_SIZE = 128
        SAVE_ITER = 1000
        EPOCHS = 90

    EMAIL_EVERY = MESSAGE_EVERY * 20
    N_CLASSES = 252
    NUM_CORES = 4

    classes = util.pkl_load(workspace.class_pkl)
    encoder = OneHot(classes)
    data = DataSet(workspace.train_pkl, workspace.test_pkl,
                   workspace.valid_pkl, workspace.class_pkl,
                   img_shape=(224, 224, 3))

    config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES)

    graph = tf.Graph()
    with graph.as_default():
        model = AlxNet(N_CLASSES, train=True, lrn=False)
        train_labels_placeholder = tf.placeholder(tf.float32, shape=None, name="train_labels")
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model.logits, train_labels_placeholder))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

        sess = tf.Session(config=config)
        with sess.as_default():
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            print "\n" + "*" * 50
            ckpt = tf.train.get_checkpoint_state(workspace.alexnet_models)
            if ckpt is not None:
                print "\nCheckpoint {0} restored!".format(os.path.basename(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print "Initialized"

            print "Training on {0} exmaples".format(len(data.tracker['train']['data']))

            print "\n", "*" * 50
            print "Batch size: {0} images".format(TRAIN_BATCH_SIZE)

            performance_data = {}
            epoch = 0
            i = 0
            try:
                while epoch <= EPOCHS:
                    start = time.time()
                    # make the data object return raw labels
                    # make the encoder encode all labels separate from
                    # the data loader
                    train_data, train_labels, epoch = data.train_batch(TRAIN_BATCH_SIZE)
                    train_lab_vec = encoder.encode(train_labels)

                    feed = {model.input_data: train_data,
                            train_labels_placeholder: train_lab_vec}
                    _, sess_loss, predictions = sess.run([optimizer, loss, model.softmax],
                                                         feed_dict=feed)

                    if ((i + 1) % MESSAGE_EVERY == 0) or (i == 0):
                        performance_data[i] = {}
                        minibatch_accuracy = util.accuracy(predictions, train_lab_vec)
                        # valid_accuracy = util.accuracy(valid_prediction.eval(), valid_lab_vec)

                        # collecting data for visualization later. Could prob use
                        # tensorboard
                        performance_data[i]['loss'] = sess_loss.mean()
                        performance_data[i]['minibatch accuracy'] = minibatch_accuracy
                        # performance_data[i]['valid accuracy'] = valid_accuracy
                        subj = 'Iteration {0} Minibatch accuracy: {1:0.2%}'.format(i+1, minibatch_accuracy)
                        msg = "\n" + "*" * 50
                        msg += '\nMinibatch loss at step {0}: {1:0.6f}\n'.format(i + 1, sess_loss.mean())
                        msg += subj + '\n'
                        # msg += "Valid accuracy: {0:0.2%}\n".format(valid_accuracy)
                        msg += 'Minibatch time: {0:0.0f} secs\n'.format(time.time() - start)
                        # msg += "Learn rate: {0}\n".format(learn_rate_)
                        msg += time.ctime()
                        print msg

                        if (((i + 1) % EMAIL_EVERY) == 0) and (EMAILING is True):
                            send_mail("dogcatcher update: " + subj, msg)
                    if ((i + 1) % SAVE_ITER) == 0:
                        saver.save(sess, os.path.join(workspace.alexnet_models, util.model_name(datetime.now())))
                        util.write_csv(performance_data, os.path.join(workspace.alexnet_models, 'performance.csv'))
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

                util.write_csv(performance_data, os.path.join(workspace.alexnet_models, 'performance.csv'))
                util.pkl_dump(encoder, os.path.join(workspace.alexnet_models, "encoder.pkl"))
                if EMAILING is True:
                    send_mail(subj, msg)


