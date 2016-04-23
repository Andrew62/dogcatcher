# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 18:34:32 2016

@author: Andrew
github: Andrew62
"""

import os
import pickle
import numpy as np
from config import workspace
from multiprocessing import Queue
from multiprocessing_transformer import MPTransformer

def pkl_dump(obj, fname):
    with open(fname, 'wb') as target:
        pickle.dump(obj, target, pickle.HIGHEST_PROTOCOL)

    
def format_array(classes, new_folders):
    print 'formatting img ref array...'
    data = []
    for folder in new_folders:
        files = filter(lambda x : x.endswith('jpg'), os.listdir(folder))
        label = os.path.basename(folder)
        for img in files:
            data.append([label, os.path.join(folder, img)])
            
    return np.asarray(data)
    
def split_data(img_data_array, splits, perms=2):
    idx = 0
    total_rows = img_data_array.shape[0]
    increment = lambda x, percent: x+int(total_rows*percent)
    
    for _ in range(perms):
        img_data_array = np.random.permutation(img_data_array)
    
    for subset, percent in splits.items():
        step = increment(idx, percent)
        splits[subset] = img_data_array[idx:step]
        idx = step
    return splits


if __name__ == "__main__":
    PICKLE_DIR = workspace.pickle_dir
    DATA_DIR = workspace.data_dir
    OUT_DIR = workspace.out_dir
    OUT_CSV = workspace.out_csv
    PROCESSES=8
    PIXELS=256
    q = Queue()
    new_folders = []
    processes = []
    classes = set()

    for i in range(PROCESSES):
        process = MPTransformer(str(i),q,PIXELS)
        processes.append(process)
        process.daemon = True
        process.start()
        
    for raw_folder in os.listdir(DATA_DIR):
        if raw_folder == '.DS_Store':
            continue
        indir = os.path.join(DATA_DIR, raw_folder)

        #Catching the one special case
        if raw_folder == 'Cirneco dell’Etna':
            raw_folder = 'Etna Cirneco'

        outdir = os.path.join(OUT_DIR, raw_folder)
        new_folders.append(outdir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        job = {'name':raw_folder, 'indir':indir, 'outdir':outdir}
        
        #collect unique image labels
        classes.add(raw_folder)
        
        q.put(job)
        
    for p in processes:
        p.join()
    
    print "makeing train, test, validation split..."
    img_data_array = format_array(classes, new_folders)
    
    data_split_percentage = {'train': 0.8, 'test': 0.1, 'valid': 0.1}
    
    data_splits = split_data(img_data_array, data_split_percentage)
    
    print "Saving..."
    for name, arr in data_splits.iteritems():
        path = os.path.join(PICKLE_DIR, "{0}.pkl".format(name))
        pkl_dump(arr, path)
        print name, arr.shape
    class_path = os.path.join(PICKLE_DIR, 'classes.pkl')
    pkl_dump(list(classes), class_path)

    print "complete!"