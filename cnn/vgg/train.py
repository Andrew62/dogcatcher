#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""

from cnn.data import DataSet
from cnn import util
import tensorflow as tf
from cnn.msg import send_mail
from cnn.encoder import OneHot
from cnn.config import workspace


ITERATIONS = 5
N_CLASSES = 252
NUM_CORES = 4
MESSAGE_EVERY = 100
EMAILING = True
EMAIL_EVERY = MESSAGE_EVERY * 20
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 400
VALID_BATCH_SIZE = 400


classes = util.pkl_load(workspace.class_pkl)
encoder = OneHot(classes)
data = DataSet(workspace.train_pkl, workspace.test_pkl,
               workspace.valid_pkl, workspace.class_pkl,
               img_shape=(256, 256, 3))

graph = tf.Graph()
with graph.as_default():
    pass



