# -*- coding: utf-8 -*-
"""
Created on Tue May 17 07:09:23 2016

@author: Andrew
github: Andrew62
"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from cnn.data import DataSet
import matplotlib.pyplot as plt
from cnn.config import workspace
from cnn.encoder import OneHot


if __name__ == "__main__":
    dat = DataSet(workspace.train_pkl, workspace.test_pkl, workspace.valid_pkl,
                  workspace.class_pkl, img_shape=(256,256,3))
    encoder = OneHot(dat.classes)
    print len(dat.classes)
    for item in ['train', 'test', 'valid']:
        print item, len(np.unique(dat.tracker[item]['data'][:,1]))
    start = time.time()
    n_iter = 10
    for i in range(n_iter):
        train, lab = dat.test_batch(20)
        lab_vec = encoder.encode(lab)
        print "Input label: {0}\n".format(lab[0]),
        #print "Decoded label: {0}".format(encoder.decode(lab_vec[0,:]))

    for i in range(n_iter):
        train, lab = dat.train_batch(20)
        lab_vec = encoder.encode(lab)
        print "Input label: {0}\n".format(lab[0]),
        #print "Decoded label: {0}".format(encoder.decode(lab_vec[0, :]))

    for i in range(n_iter):
        train, lab = dat.valid_batch(20)
        lab_vec = encoder.encode(lab)
        print "Input label: {0}\n".format(lab[0]),
        #print "Decoded label: {0}".format(encoder.decode(lab_vec[0, :]))

    elapsed = time.time() - start
    print "Complete in {0:0.2f} seconds".format(elapsed)
    print "Average batch load {0:0.4f}".format(elapsed/(n_iter*1.))


    
    hist, bins = np.histogram(train, bins=100)
    width = .7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:])/2
    plt.imshow(train[1,:,:,:])
    plt.show()
