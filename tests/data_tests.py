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
    dat = DataSet(workspace.test_pkl, 32, 3, img_shape=(224,224,3))
    dat.start()

    start = time.time()
    epoch = 0
    n_iter = 0
    while epoch < 1:
        train, lab, epoch = dat.batch()
        print lab[0]
        n_iter += 1

    elapsed = time.time() - start
    print "Complete in {0:0.2f} seconds".format(elapsed)
    print "Average batch load {0:0.4f}".format(elapsed/(n_iter*1.))


    f, (ax1, ax2) = plt.subplots(1, 2)
    #will perform mean subtraction in network
    hist, bins = np.histogram(train[1,:,:,:], bins=100)
    width = .7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:])/2
    ax1.imshow(train[1,:,:,:])
    ax1.set_title(lab[0])
    ax2.bar(center, hist, align='center', width=width)
    ax2.set_title("Example pixel distribution")
    plt.show()
