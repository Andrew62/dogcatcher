# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 12:46:58 2016

@author: Andrew
github: Andrew62

This class is designed load image data from disk when 
the batch methods are called saving memory but increasing
batch load time. The pickle files passed to the object 
are simply tuples of (label, file-path). The class 
keeps track of the index for each data subset so it knows
when to reshuffle. Testing the load times below on a 4 core 
laptop with 8gb ram and ssd, a batch of 256 images loads 
in 25.3 seconds.
"""

import time
import pickle
import threading
import numpy as np
from PIL import Image
from Queue import Queue


class DataSet(object):

    def __init__(self, data_pkl, batch_size, **kwargs):
        self.data = self.pkl_load(data_pkl)
        self.img_shape = kwargs.pop('img_shape', (224, 224, 3))
        self.epoch = 0
        self.input_queue = Queue()
        self.output_queue = Queue(maxsize=8)
        self.batch_size = batch_size

        self.workers = []
        self._load_queues()

    def pkl_load(self, fp):
        with open(fp, 'rb') as infile:
            return np.random.permutation(pickle.load(infile))

    def _load_queues(self):
        n_workers = 4

        batch_shape = (self.batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2])

        data = np.random.permutation(self.data)

        for idx in xrange(0, data.shape[0], self.batch_size):
            self.input_queue.put(data[idx:idx + self.batch_size, :].copy())

        for _ in range(n_workers):
            self.input_queue.put(None)

        for idx in range(n_workers):
            worker = Loader(str(idx + 1), self.input_queue, self.output_queue, batch_shape)
            worker.start()
            self.workers.append(worker)


    def batch(self):
        """
        need to just shuffle then return batch size not keep
        track of current idx
        """
        if self.output_queue.empty():
            print "Empty"
            self._load_queues()
            self.epoch += 1
        batch_data, batch_labels = self.output_queue.get()
        return batch_data, batch_labels, self.epoch




class Loader(threading.Thread):

    def __init__(self, name, in_q, out_q, batch_shape):
        super(Loader, self).__init__()
        self.name = name
        self.in_q = in_q
        self.out_q = out_q
        self.batch_shape = batch_shape

    def run(self):
        while True:
            batch_data = np.ones(shape=self.batch_shape, dtype=np.float32)
            batch_labels = []
            batch = self.in_q.get(timeout=5)
            if batch is None:
                return
            for idx in xrange(batch.shape[0]):
                row = batch[idx, :]
                batch_data[idx, :, :, :] = np.array(Image.open(row[1]))
                batch_labels.append(row[0])
            self.out_q.put((batch_data, batch_labels))

