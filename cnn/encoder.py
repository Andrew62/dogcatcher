# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 11:51:23 2016

@author: Andrew
github: Andrew62

Change to accept a label array and return the encoding
so it will be separate from the data loader
"""

import numpy as np

class OneHotError(Exception):
    pass

class OneHot(object):
    def __init__(self, classes):
        #remove class duplicates
        self.classes = list(set(classes))
        self.n_classes = len(self.classes)

    @property
    def vector_template(self):
        return np.zeros(shape=(1,self.n_classes), dtype=np.float32)
        
    @property
    def class_map(self):
        cls_map = {}
        for i,cls in enumerate(self.classes):
            cls_map[cls]= i
        return cls_map
        
    @property
    def reverse_class_map(self):
        rev_class_map = {}
        for key, val in self.class_map.items():
            rev_class_map[val] = key
        return rev_class_map
        

    def decode(self, arr, top=10):
        decoded = {}
        for i in range(top):
            idx = np.argmax(arr)
            decoded[i+1] = {'score': arr[idx], 'label':self.reverse_class_map[idx]}
            arr[idx] = 0
        return decoded
            
        
    
    def encode(self, category):
        if category not in self.class_map.keys():
            raise OneHotError("{0} outside of original classes!".format(category))
        vec = self.vector_template.copy()
        vec[0, self.class_map[category]] = 1
        return vec
        

        
        
        