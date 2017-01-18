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
when to reshuffle.
"""

import pickle
import traceback
import threading
import numpy as np
from PIL import Image
from queue import Queue


def pkl_load(fp):
    """
    Load and permuatate a pickled numpy array
    :param fp: path to pickle file
    :return: permutated array
    """
    with open(fp, 'rb') as infile:
        return np.random.permutation(pickle.load(infile))


class DataSet(object):

    def __init__(self, data, batch_size, epochs, **kwargs):
        """
        Multithreaded data loader
        :param data: numpy array with rows [label, file_path]
        :param batch_size: number of examples per batch
        :param epochs: number of times to travse the data
        :param img_shape: tuple of output data. For example (224, 224, 3)
        :param n_loaders: number of data loading threads. These read files into the array
        """
        self.data = data
        self.img_shape = kwargs.pop('img_shape', (224, 224, 3))
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_workers = kwargs.pop("n_loaders", 4)

        self.input_queue = Queue(maxsize=16)
        self.output_queue = Queue(maxsize=16)
        self.workers = []
        self._event = threading.Event()

    def start(self):
        batch_shape = (self.batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2])

        q_loader = QueueLoader("queue_loader", self.input_queue, self.data, self.epochs, self.batch_size, self._event)
        q_loader.setDaemon(True)
        q_loader.start()
        self.workers.append(q_loader)
        for idx in range(self.n_workers):
            worker = BatchLoader(str(idx + 1), self.input_queue, self.output_queue, batch_shape, self._event)
            worker.setDaemon(True)
            worker.start()
            self.workers.append(worker)

    def stop(self):
        self._event.set()

    def __del__(self):
        self.stop()

    def batch(self):
        """
        :return: batch_data, batch_labels, epoch
        """
        batch_data, batch_labels, epoch = self.output_queue.get()
        if self.epochs == epoch:
            self.stop()
        return batch_data, batch_labels, epoch


class QueueLoader(threading.Thread):
    def __init__(self, name, queue, data, epochs, batch_size, event):
        """
        Queue to create batches of input data. Does not load data just
        splits an input array and puts it in a queue
        :param name: object name
        :param queue: queue object
        :param data: array to divide up
        :param epochs: number of times to go through the data
        :param batch_size: elements per batch
        :param event: threading.Event object
        """
        super(QueueLoader, self).__init__()
        self.name = name
        self.queue = queue
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self._stop = event

    def run(self):
        try:
            print(self.name + " starting")
            epoch = 0
            while epoch < self.epochs and not self._stop.is_set():
                data = np.random.permutation(self.data)
                for idx in range(0, data.shape[0], self.batch_size):
                    batch = data[idx:idx + self.batch_size, :].copy()
                    if (idx + self.batch_size) > data.shape[0]:
                        continue
                    self.queue.put((epoch, batch))
                epoch += 1
        except Exception as e:
            traceback.print_exc()
            raise e

        print(self.name + " stopping")


class BatchLoader(threading.Thread):

    def __init__(self, name, in_q, out_q, batch_shape, event):
        """
        Object to read data using an array to specify labels and
        file paths
        :param name: thread name
        :param in_q: Queue object with array batches
        :param out_q: Queue to hold the processed batches. DataSet object
            uses this queue to get data into the main process
        :param batch_shape: 4 tuple of (batch_size, height, width, channels
        :param event: threading.Event object
        """
        super(BatchLoader, self).__init__()
        self.name = name
        self.in_q = in_q
        self.out_q = out_q
        self.batch_shape = batch_shape

        self._stop = event

    @staticmethod
    def normalize(img):
        return (img - img.mean())/img.std()

    def run(self):
        try:
            print(self.name + " starting")
            while not self._stop.is_set():
                batch_data = np.ones(shape=self.batch_shape, dtype=np.float32)
                batch_labels = []
                epoch, batch = self.in_q.get()
                for idx in range(batch.shape[0]):
                    row = batch[idx, :]
                    batch_data[idx, :, :, :] = np.array(Image.open(row[1]))
                    batch_labels.append(row[0])
                self.out_q.put((batch_data, batch_labels, epoch))
        except Exception as e:
            traceback.print_exc()
            raise e

