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
                  workspace.class_pkl, img_shape=(224,224,3))
    encoder = OneHot(dat.classes)
    print len(dat.classes)
    for item in ['train', 'test', 'valid']:
        print item, len(np.unique(dat.tracker[item]['data'][:,1]))
    start = time.time()
    n_iter = 10
    epoch = 0
    iter= 0
    while epoch < 1:
        train, lab, epoch = dat.test_batch(20)
        lab_vec = encoder.encode(lab)
        print "Input label: {0}\n".format(lab[0]),
        #print "Decoded label: {0}".format(encoder.decode(lab_vec[0,:]))
        if iter > 10:
            break
        iter += 1

    for i in range(n_iter):
        train, lab, epoch = dat.train_batch(20)
        lab_vec = encoder.encode(lab)
        print "Input label: {0}\n".format(lab[0]),
        #print "Decoded label: {0}".format(encoder.decode(lab_vec[0, :]))

    for i in range(n_iter):
        train, lab, epoch = dat.valid_batch(20)
        lab_vec = encoder.encode(lab)
        print "Input label: {0}\n".format(lab[0]),
        #print "Decoded label: {0}".format(encoder.decode(lab_vec[0, :]))

    elapsed = time.time() - start
    print "Complete in {0:0.2f} seconds".format(elapsed)
    print "Average batch load {0:0.4f}".format(elapsed/(n_iter*1.))


    f, (ax1, ax2) = plt.subplots(1, 2)
    #will perform mean subtraction in network
    hist, bins = np.histogram(train-np.mean(train, axis=0), bins=100)
    width = .7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:])/2
    ax1.imshow(train[1,:,:,:])
    ax1.set_title(lab[0])
    ax2.bar(center, hist, align='center', width=width)
    ax2.set_title("Example pixel distribution")
    plt.show()
