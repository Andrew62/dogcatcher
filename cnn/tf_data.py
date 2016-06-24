#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""
from __future__ import division

import numpy as np
import tensorflow as tf

class ImageProducer(object):

    def __init__(self, img_paths, labels, **kwargs):
        """
        A class for loading input images. Images will be transformed
        to the specified img_shape.

        :param img_paths: an iterable containing jpg paths
        :param labels: an iterable with string labels
        :param n_processes: number of threads to use to run enqueue data. default 4
        :param batch_size: number of samples to load. default 128
        :param img_shape: shape of the input image. Default is (224, 224, 3)
        """

        self.img_paths = img_paths
        self.labels = labels
        self.n_processes = kwargs.pop('n_processes', 4)
        self.batch_size = kwargs.pop('batch_size', 128)
        self.img_shape = kwargs.pop("img_shape", (224, 224, 3))

        self.data_queue = tf.RandomShuffleQueue(capacity=self.num_images, min_after_dequeue=1,
                                                dtypes=[tf.string, tf.string],
                                                name="data_queue")
        self.data_enqueue_op = self.data_queue.enqueue_many([self.labels, self.img_paths])
        self.closed_queue = self.data_queue.close()

        label, processed_image = self.process()

        process_queue = tf.FIFOQueue(capacity=np.ceil(self.num_images/self.n_processes),
                                     dtypes=[tf.string, tf.float32],
                                     shapes=[(), self.img_shape])
        enqueue_process = process_queue.enqueue([label, processed_image])

        self.dequeue_op = process_queue.dequeue_many(self.batch_size)

        n_processes = min(self.n_processes, self.num_images)
        self.queue_runner = tf.train.QueueRunner(process_queue, [enqueue_process] * n_processes)

    def process(self):
        label, img_path = self.data_queue.dequeue()
        img = self.load_image(img_path)
        resized = tf.image.resize_images(img, self.img_shape[0], self.img_shape[1])
        return label, tf.to_float(resized)

    def load_image(self, img_path):
        raw_data = tf.read_file(img_path)
        return tf.image.decode_jpeg(raw_data, channels=self.img_shape[-1])

    def start(self, sess, coord):
        """
        Starts the queue runners

        :param sess: tensorflow session
        :param coord: tensorflow coordinator
        :return: queue runner threads that need to be joined by a coordinator
        """
        sess.run(self.data_enqueue_op)
        sess.run(self.closed_queue)
        return self.queue_runner.create_threads(sess, coord=coord, start=True)

    def get_batch(self, sess):
        """
        Loads a batch of data

        :param sess: a tensorflow session
        :return: lables, images
        """
        labels, images = sess.run(self.dequeue_op)
        return labels, images

    @property
    def num_images(self):
        return len(self.img_paths)

