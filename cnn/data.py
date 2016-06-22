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
    def __init__(self, train_pkl, test_pkl, valid_pkl, class_pkl, **kwargs):
        self.tracker = {'train':{}, 'test':{}, 'valid':{}}
        self.tracker['train']['data'] = self.pkl_load(train_pkl)
        self.tracker['test']['data'] = self.pkl_load(test_pkl)
        self.tracker['valid']['data'] = self.pkl_load(valid_pkl)
        self.classes = self.pkl_load(class_pkl)
        self.img_shape = kwargs.pop('img_shape', (224,224,3))
        
        for item in ['train', 'test', 'valid']:
            self.tracker[item]['idx'] = 0
            self.tracker[item]['epoch'] = 0

    def pkl_load(self, fp):
        with open(fp, 'rb') as infile:
            return np.random.permutation(pickle.load(infile))
            
    @property
    def n_classes(self):
        return len(self.classes)
        
    def batch(self, batch_size, name='train'):
        """
        need to just shuffle then return batch size not keep
        track of current idx
        """
        stop = self.tracker[name]['idx'] + batch_size
        
        if stop > self.tracker[name]['data'].shape[0]:
            # If the slice goes beyond the number of rows, shuffle the whole
            # thing and start over
            print '\n', "*" * 50
            print "*" * 50
            print "Shuffling {0} data...".format(name)
            print "*" * 50
            print '\n', "*" * 50

            self.tracker[name]['data'] = np.random.permutation(self.tracker[name]['data'])
            self.tracker[name]['idx'] = 0
            self.tracker[name]['epoch'] += 1
            
        batch_shape = (batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2])
        batch_data = np.ones(shape=batch_shape, dtype=np.float32)
        batch_labels = []
        for i in range(batch_size):
            idx = self.tracker[name]['idx'] + i
            row = self.tracker[name]['data'][idx, :]
            batch_data[i,:,:,:] = imread(row[1])
            batch_labels.append(row[0])
        self.tracker[name]['idx'] += batch_size
        return batch_data, batch_labels
        
    def train_batch(self, batch_size):
        name = 'train'
        samples, labels  = self.batch(batch_size, name)
        return samples, labels, self.tracker[name]['epoch']
        
    def test_batch(self, batch_size):
        name='test'
        samples, labels = self.batch(batch_size, name)
        return samples, labels, self.tracker[name]['epoch']

    def valid_batch(self, batch_size):
        name = 'valid'
        samples, labels = self.batch(batch_size, name)
        return samples, labels, self.tracker[name]['epoch']
    

            
        
        
        

    
