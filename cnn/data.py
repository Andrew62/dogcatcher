# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 12:46:58 2016

@author: Andrew
github: Andrew62
"""

import pickle
import numpy as np
from encoder import OneHot
from scipy.misc import imread



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

    def pkl_load(self, fp):
        with open(fp, 'rb') as infile:
            return pickle.load(infile)
            
    @property
    def n_classes(self):
        return len(self.classes)
                
    @property
    def encoder(self):
        return OneHot(self.classes)
        
    def batch(self, batch_size, name='train'):
        """
        need to just shuffle then return batch size not keep
        track of current idx
        """
        stop = self.tracker[name]['idx'] + batch_size
        
        if stop > self.tracker[name]['data'].shape[0]:
            #If the slice goes beyond the number of rows, shuffle the whole 
            #thing and start over
            print "Shuffling {0} data...".format(name)
            self.tracker[name]['data'] = np.random.permutation(self.tracker[name]['data'])
            self.tracker[name]['idx'] = 0
            
        batch_shape = (batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2])
        batch_data = np.zeros(shape=batch_shape, dtype=np.float32)
        batch_labels = np.zeros(shape=(batch_size,len(self.classes)), dtype=np.float32)
        for i in range(batch_size):
            idx = self.tracker[name]['idx'] + 1
            row = self.tracker[name]['data'][idx, :]
            img = imread(row[1])
            label = self.encoder.encode(row[0])
            batch_data[i,:,:,:] = img.astype(np.float32)
            batch_labels[i,:] = label
        self.tracker[name]['idx'] += batch_size
        normed = (batch_data - np.mean(batch_data))/(np.std(batch_data))
        return normed, batch_labels
        
    def train_batch(self, batch_size):
        return self.batch(batch_size, 'train')
        
    def test_batch(self, batch_size):
        return self.batch(batch_size, 'test')
        
    def valid_batch(self, batch_size):
        return self.batch(batch_size, 'valid')
    

            
        
        
        
if __name__ == "__main__":
    import time
    from config import workspace
    import matplotlib.pyplot as plt
    
    dat = DataSet(workspace.train_pkl, workspace.test_pkl, workspace.valid_pkl,
                  workspace.class_pkl, img_shape=(256,256,3))
    print len(dat.classes)
    for item in ['train', 'test', 'valid']:
        print item, len(np.unique(dat.tracker[item]['data'][:,1]))
    start = time.time()
    n_iter = 10
    for i in range(n_iter):
        train, lab = dat.test_batch(25)
        if (i+1)%100 == 0:
            print i+1
    elapsed = time.time() - start
    print "Complete in {0:0.2f} seconds".format(elapsed)
    print "Average batch load {0:0.4f}".format(elapsed/(n_iter*1.))
    print dat.encoder.decode(lab[1,:], 1)
#    plt.imshow(train[1,:,:,:])
#    plt.show()

    
