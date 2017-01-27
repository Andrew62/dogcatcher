#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""

import os
import util
import tensorflow as tf
from nets import inception
from config import workspace
from datatf import batch_producer
from datasets import dataset_utils

slim = tf.contrib.slim

debug = True
        
if debug is True:
    print("DEBUG MODE")
    BATCH_SIZE = 2
    EPOCHS = 1
else:
    BATCH_SIZE = 16
    EPOCHS = 90


classes = util.pkl_load(workspace.class_pkl)
csv_files = ["/Users/awoizesko/Documents/dogcatcher_Project/dogcatcher/tests/test_data/dogs/dogs.csv"]

if not tf.gfile.Exists(workspace.inception_cpkt):
    tf.gfile.MakeDirs(workspace.inception_cpkt)

if not os.path.exists(os.path.join(workspace.inception_cpkt, os.path.basename(workspace.inception_url))):
    dataset_utils.download_and_uncompress_tarball(workspace.inception_url,
                                                  workspace.inception_cpkt)


def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = ["InceptionV1/Logits", "InceptionV1/AuxLogits"]

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        os.path.join(workspace.inception_cpkt, 'inception_v1.ckpt'),
        variables_to_restore)

graph = tf.Graph()
with graph.as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    labels, images = batch_producer(csv_files, len(classes),
                                    batch_size=BATCH_SIZE, epochs=EPOCHS,
                                    img_shape=workspace.img_shape,
                                    num_threads=2)

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(images, num_classes=len(classes),
                                           is_training=True)

    # class probabilities. Only used at the end to see accuracy metrics
    probabilities = tf.nn.softmax(logits)

    # losses
    slim.losses.softmax_cross_entropy(logits, labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/Total_Loss', total_loss)

    optimizer = tf.train.AdamOptimizer(0.01)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # starts a training loop
    final_loss = slim.learning.train(
        train_op,
        logdir=os.path.join(workspace.inception_cpkt, "log"),
        init_fn=get_init_fn())

print("Final loss: {0:0.2f}".format(final_loss))
