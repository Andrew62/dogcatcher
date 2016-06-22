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
        TRAIN_BATCH_SIZE = 32
        SAVE_ITER = 30
        EPOCHS = 1
    else:
        MESSAGE_EVERY = 25
        EMAILING = True
        TRAIN_BATCH_SIZE = 128
        SAVE_ITER = 1000
        EPOCHS = 90

    EMAIL_EVERY = MESSAGE_EVERY * 80
    n_classes = 252
    keep_prob = 0.5
    train = True

    classes = util.pkl_load(workspace.class_pkl)
    encoder = OneHot(classes)
    data = DataSet(workspace.train_pkl, workspace.test_pkl,
                   workspace.valid_pkl, workspace.class_pkl,
                   img_shape=(224, 224, 3))

    graph = tf.Graph()
    with graph.as_default():
        input_data_placeholder = tf.placeholder(dtype=tf.float32, name="input_data")#, shape=(128, 224, 224, 3))
        train_labels_placeholder = tf.placeholder(dtype=tf.float32, name="train_labels")
        keep_prob = tf.constant(keep_prob, name="Dropout", dtype=tf.float32)
        tf.image_summary('input_data_placeholder', input_data_placeholder, max_images=1)

        with tf.variable_scope('batch_norm'):
            mean, var = tf.nn.moments(input_data_placeholder, axes=[0, 1, 2])
            batch_norm = tf.nn.batch_normalization(input_data_placeholder, mean, var, offset=None, scale=None,
                                                   variance_epsilon=1e4)
            tf.image_summary("batch_norm", batch_norm, max_images=1)
            helper.var_summary(batch_norm, "batch_norm")

        with tf.variable_scope("pool1"):
            with tf.variable_scope('conv1'):
                weights1 = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32, stddev=1e-2))
                bias1 = tf.Variable(tf.constant(1.0, shape=[96], dtype=tf.float32))
                conv1 = tf.nn.conv2d(batch_norm, weights1, [1, 4, 4, 1], 'SAME')
                hidden1 = tf.nn.relu(conv1 + bias1)
                helper.var_summary(hidden1, 'conv1')
                response_norm1 = tf.nn.local_response_normalization(conv1, depth_radius=5, alpha=1e-3,
                                                                    beta=0.75, bias=2.0)
                helper.var_summary(response_norm1, 'response_norm1')

            pool1 = tf.nn.max_pool(response_norm1, [1, 2, 2, 1], [1, 1, 1, 1], padding="VALID")
            helper.var_summary(pool1, 'pool1')

        with tf.variable_scope("pool2"):
            with tf.variable_scope('conv2'):
                weights2 = tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=1e-2, dtype=tf.float32))
                bias2 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[256]))
                conv2 = tf.nn.conv2d(pool1, weights2, [1, 1, 1, 1], 'SAME')
                hidden2 = tf.nn.relu(conv2 + bias2)
                helper.var_summary(hidden2, 'conv2')
                response_norm2 = tf.nn.local_response_normalization(hidden2, depth_radius=5, alpha=1e-3,
                                                                beta=0.75, bias=2.0)
                helper.var_summary(response_norm2, "reponse_norm2")

            pool2 = tf.nn.max_pool(response_norm2, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")
            helper.var_summary(pool2, 'pool2')

        with tf.variable_scope("pool3"):
            with tf.variable_scope('conv3'):
                weights3 = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32, stddev=1e-2))
                bias3 = tf.Variable(tf.constant(1.0, shape=[384], dtype=tf.float32))
                conv3 = tf.nn.conv2d(pool2, weights3, [1, 2, 2, 1], 'VALID')
                hidden3 = tf.nn.relu(conv3 + bias3)
                helper.var_summary(hidden3, 'conv3')

            with tf.variable_scope("conv4"):
                weights4 = tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=1e-2, dtype=tf.float32))
                bias4 = tf.Variable(tf.constant(1.0, shape=[384], dtype=tf.float32))
                conv4 = tf.nn.conv2d(hidden3, weights4, [1, 1, 1, 1], 'SAME')
                hidden4 = tf.nn.relu(conv4 + bias4)
                helper.var_summary(hidden4, 'conv4')

            with tf.variable_scope("conv5"):
                weights5 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=1e-2, dtype=tf.float32))
                bias5 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[256]))
                conv5 = tf.nn.conv2d(hidden4, weights5, [1, 1, 1, 1], 'SAME')
                hidden5 = tf.nn.relu(conv5 + bias5)
                helper.var_summary(hidden5, 'conv5')

            pool5 = tf.nn.max_pool(hidden5, [1, 2, 2, 1], [1, 1, 1, 1], padding="SAME")
            helper.var_summary(pool5, 'pool5')

            middle_shape = 43264

            reshape5 = tf.reshape(pool5, [-1, middle_shape])

        with tf.variable_scope("fc6"):
            weights6 = tf.Variable(tf.truncated_normal([middle_shape, 4096], dtype=tf.float32, stddev=1e-2))
            bias6 = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32))
            matmul_1 = tf.matmul(reshape5, weights6)
            fc6 = tf.nn.relu(matmul_1 + bias6)
            helper.var_summary(fc6, 'fc6')

            if train is True:
                fc6 = tf.nn.dropout(fc6, keep_prob)

        with tf.variable_scope("fc7"):
            weights7 = tf.Variable(tf.truncated_normal([4096, 4096], stddev=1e-2, dtype=tf.float32))
            bias7 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[4096]))
            matmul_2 = tf.matmul(fc6, weights7)
            fc7 = tf.nn.relu(matmul_2 + bias7)
            helper.var_summary(fc7, 'fc7')

            if train is True:
                fc7 = tf.nn.dropout(fc7, keep_prob)

        with tf.variable_scope("logits"):
            weights8 = tf.Variable(tf.truncated_normal([4096, n_classes], stddev=1e-2, dtype=tf.float32))
            bias8 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[n_classes]))
            logits = tf.matmul(fc7, weights8) + bias8
            helper.var_summary(logits, 'logits')

        softmax = tf.nn.softmax(logits, 'softmax')

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_placeholder))
        tf.scalar_summary('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

        saver = tf.train.Saver()
        initop = tf.initialize_all_variables()
        merged = tf.merge_all_summaries()


    sess = tf.Session(graph=graph)
    summary_writer = tf.train.SummaryWriter(workspace.alexnet_summary, graph=graph)
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
        print "Batch size: {0} images".format(TRAIN_BATCH_SIZE)
        epoch = 0
        i = 0
        try:
            while epoch <= EPOCHS:
                start = time.time()
                train_data, train_labels, epoch = data.train_batch(TRAIN_BATCH_SIZE)
                train_lab_vec = encoder.encode(train_labels)

                feed = {input_data_placeholder: train_data,
                        train_labels_placeholder: train_lab_vec}
                _, sess_loss, predictions, summary = sess.run([optimizer, loss, softmax, merged],
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