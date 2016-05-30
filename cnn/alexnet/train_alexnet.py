"""
This module will be used to run training sessions. Code will be refactored so
different image classifiers can be used easily.
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 16:03:40 2016

@author: Andrew
github: Andrew62

"""

import os
import time
import util
import shutil
import tensorflow as tf
from data import DataSet
from msg import send_mail
from encoder import OneHot
from alexnet import  AlxNet
from config import workspace
from datetime import datetime
from wrapper import constant, placeholder


# need tensorflow_serving to run model
# intall http://tensorflow.github.io/serving/setup
# see http://tensorflow.github.io/serving/serving_basic
# need to download bazel
# https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md
# from tensorflow_serving.session_bundle import exporter


data = DataSet(workspace.train_pkl, workspace.test_pkl,
               workspace.valid_pkl, workspace.class_pkl,
               img_shape=(256, 256, 3))



ITERATIONS = 50001
SAVE_ITER = 1000
NUM_CORES = 4
MESSAGE_EVERY = 100
EMAILING = True
EMAIL_EVERY = MESSAGE_EVERY * 20
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 400
VALID_BATCH_SIZE = 400
N_CLASSES = data.n_classes

onehot = OneHot(data.classes)

test_data, test_labels = data.test_batch(TEST_BATCH_SIZE)
test_lab_vec = onehot.encode(test_labels)
valid_data, valid_labels = data.valid_batch(VALID_BATCH_SIZE)
valid_lab_vec = onehot.encode(valid_labels)


config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                        intra_op_parallelism_threads=NUM_CORES)

graph = tf.Graph()
with graph.as_default():
    train_data_placeholder = placeholder("train_data_placeholder")
    train_labels_placeholder = placeholder('train_labels_placeholder')
    valid_data_placeholder = constant(valid_data, "valid_data_placeholder")
    tf_test_data = constant(test_data, "test_data_placehoder")

    model = AlxNet(N_CLASSES)

    logits = model.predict(train_data_placeholder)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_placeholder))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model.predict(valid_data_placeholder))
    test_prediction = tf.nn.softmax(model.predict(tf_test_data))

    sess = tf.Session(config=config)
    with sess.as_default():
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        print "\n" + "*" * 50
        ckpt = tf.train.get_checkpoint_state(workspace.model_dir)
        if ckpt is not None:
            print "\nCheckpoint {0} restored!".format(os.path.basename(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "Initialized"

        print "\n", "*" * 50
        print "Batch size: {0} images".format(TRAIN_BATCH_SIZE)

        performance_data = {}
        try:
            for i in xrange(ITERATIONS):
                performance_data[i] = {}
                start = time.time()
                # make the data object return raw labels
                # make the encoder encode all labels separate from
                # the data loader
                train_data, train_labels = data.train_batch(TRAIN_BATCH_SIZE)
                train_lab_vec = onehot.encode(train_labels)

                feed_dict = {train_data_placeholder: train_data,
                             train_labels_placeholder: train_lab_vec,}
                _, sess_loss, predictions = sess.run([optimizer, loss, train_prediction],
                                                     feed_dict=feed_dict)

                if ((i + 1) % MESSAGE_EVERY == 0) or (i == 0):
                    minibatch_accuracy = util.accuracy(predictions, train_lab_vec)
                    valid_accuracy = util.accuracy(valid_prediction.eval(), valid_lab_vec)

                    # collecting data for visualization later. Could prob use
                    # tensorboard
                    performance_data[i]['loss'] = sess_loss.mean()
                    performance_data[i]['minibatch accuracy'] = minibatch_accuracy
                    performance_data[i]['valid accuracy'] = valid_accuracy
                    subj = 'Minibatch accuracy: {0:0.2%}'.format(minibatch_accuracy)
                    msg = "\n" + "*" * 50
                    msg += '\nMinibatch loss at step {0}: {1:0.6f}\n'.format(i + 1, sess_loss.mean())
                    msg += subj + '\n'
                    msg += "Valid accuracy: {0:0.2%}\n".format(valid_accuracy)
                    msg += 'Minibatch time: {0:0.0f} secs\n'.format(time.time() - start)
                    msg += time.ctime()
                    print msg
                    if (((i + 1) % EMAIL_EVERY) == 0) and (EMAILING is True):
                        send_mail("dogcatcher update: " + subj, msg)
                if ((i + 1) % SAVE_ITER) == 0:
                    saver.save(sess, os.path.join(workspace.model_dir, util.model_name(datetime.now())))
                    if EMAILING is True:
                        send_mail("Successful checkpoint", "Iteration {0}".format(i + 1))
            msg = "\n" + "*" * 50
            msg += "\n" + "*" * 50
            msg += "\nTest accuracy: {0:0.2%}".format(util.accuracy(test_prediction.eval(), test_lab_vec))
            subj = "Training complete!"
            print msg
        except Exception as e:
            print e
            subj = "DOGCATCHER STOPPED!"
            msg = "Failed after {0} steps".format(i)
            print msg

        finally:
            saver.save(sess, os.path.join(workspace.model_dir, util.model_name(datetime.now())))
            outg = os.path.join(workspace.model_dir, "graph")
            if os.path.exists(outg):
                shutil.rmtree(outg)
            tf.train.write_graph(sess.graph_def, outg, "graph.pb")

            util.write_csv(performance_data, os.path.join(workspace.model_dir, 'performance.csv'))
            util.pkl_dump(onehot, os.path.join(workspace.model_dir, "encoder.pkl"))
            if EMAILING is True:
                send_mail(subj, msg)