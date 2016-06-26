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

import os
import pickle
import numpy as np
from skimage.io import imread



class DataSet(object):
    def __init__(self, data_pkl, **kwargs):
        self.data = self.pkl_load(data_pkl)
        self.img_shape = kwargs.pop('img_shape', (224,224,3))
        self.idx = 0
        self.epoch = 0

    def pkl_load(self, fp):
        with open(fp, 'rb') as infile:
            return np.random.permutation(pickle.load(infile))

    def batch(self, batch_size):
        """
        need to just shuffle then return batch size not keep
        track of current idx
        """
        stop = self.idx + batch_size
        
        if stop > self.data.shape[0]:
            # If the slice goes beyond the number of rows, shuffle the whole
            # thing and start over
            print '\n', "*" * 50
            print "*" * 50
            print "Shuffling data..."
            print "*" * 50
            print '\n', "*" * 50

            self.data = np.random.permutation(self.data)
            self.idx = 0
            self.epoch += 1
            
        batch_shape = (batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2])
        batch_data = np.ones(shape=batch_shape, dtype=np.float32)
        batch_labels = []
        for i in range(batch_size):
            idx = self.idx + i
            row = self.data[idx, :]
            batch_data[i,:,:,:] = imread(row[1])
            batch_labels.append(row[0])
        self.idx += batch_size
        return batch_data, batch_labels, self.epoch