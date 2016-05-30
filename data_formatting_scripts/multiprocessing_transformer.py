# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 11:18:33 2016

@author: Andrew
github: Andrew62
"""

import os
import time
import pickle
import hashlib
import numpy as np
from scipy import misc
from random import randint
import skimage.transform as tf
from skimage.data import imread
from multiprocessing import Process
           

class MPTransformer(Process):
    def __init__(self, name, queue, pixels=256):
        Process.__init__(self)
        self.queue = queue
        self.name = name
        self.pixels = pixels   
        
    def reshape(self, a, b):
        """
        returns the scaled image size and eventual slices
        """
        scale_factor = (self.pixels*1.)/a
        new_b = int(np.ceil(scale_factor * b))
        new_a = self.pixels
        new_a_slice = [0,self.pixels]
        b_diff = np.ceil((new_b-self.pixels)/2.).astype(np.int64)
        new_b_slice = [b_diff, new_b-b_diff]
        return new_a, new_b, new_a_slice, new_b_slice
        
    def image_center(self, image):
        """
        reshapes, resizes, and returns a means subtracted image
        """
        rows, cols, layers = image.shape
        if rows != cols:
            if rows < cols:
                new_rows, new_cols, new_rows_slice, new_cols_slice = self.reshape(rows, cols)
                
            elif rows > cols:
                new_cols, new_rows, new_cols_slice, new_rows_slice = self.reshape(cols, rows)
                
            resized_image = misc.imresize(image, (new_rows, new_cols, 3))
            image_slice = resized_image[new_rows_slice[0]:new_rows_slice[1],new_cols_slice[0]:new_cols_slice[1],:]
            output = misc.imresize(image_slice, (self.pixels,self.pixels, 3))
            
        else:
            new_rows = self.pixels
            new_cols = self.pixels
            output = misc.imresize(image, (new_rows, new_cols, 3))
        return ('centered', output)
        
    def rotate(self, image, degrees):
        shift_y, shift_x = np.array(image.shape[:2]) / 2.
        tf_rotate = tf.SimilarityTransform(rotation=np.deg2rad(degrees))
        tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])
        image_rotated = tf.warp(image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
        __, img = self.image_center(image_rotated)
        return ('rotate_{0}'.format(degrees), img)
        
    def flip(self, image):
        return ('flipped', np.fliplr(image))
    
    def save_hashes(self, outdir, name, img_hashes):
        with open(os.path.join(outdir, "{0}.pkl".format(name)), 'wb') as target:
            pickle.dump(img_hashes, target, pickle.HIGHEST_PROTOCOL)
    
    def load_hashes(self, indir):
        with open(indir, 'rb') as infile:
            return pickle.load(infile)
            
    def scale(self, image, layers=3):
        """
        Normalized all channels between 0 and 1
        """
        image = image.astype(np.float32)
        for channel in range(layers):
            image[:,:,channel] = (image[:,:,channel] - np.mean(image[:,:,channel]))/np.std(image[:,:,channel])

        return image
        
    def processed_hash(self, outdir, name):
        filep = os.path.join(outdir, "{0}.pkl")
        if os.path.exists(filep):
            return self.load_hashes(filep)
        return set()
            
        
    def dedup(self, indir, outdir, name):
        duplicate_counter = 0
        already_used = self.processed_hash(outdir, name)
        img_files = os.listdir(indir)
        valid_files = []
        for img_file in img_files:
            if img_file == '.DS_Store':
                continue
            try:
                img_path = os.path.join(indir,img_file)
                img = imread(img_path)
                img_hash = hashlib.sha1(img).digest()
                if img_hash in already_used:
                    duplicate_counter+= 1
                    continue
                already_used.add(img_hash)
                rows, cols, layers = img.shape
                if layers != 3:
                    continue
                valid_files.append(img_path)
            except IOError:
                continue
            except ValueError:
                continue
        self.save_hashes(outdir, name, already_used)
        return valid_files, duplicate_counter, len(img_files)
        
    def create_blank_array(self, valid_files):
        shape=(5*len(valid_files), self.pixels, self.pixels, 3)
        return np.zeros(shape=shape, dtype=np.uint8)
    
    def save_img(self, img, outdir, name, index, trans):
        name = "{0}_{1}_{2}.jpg".format(name, trans, index)
        path = os.path.join(outdir, name)
        misc.imsave(path, img)
        
    def transform(self, valid_files, outdir, name):
        success = 0
        for img_file in valid_files:
            img = imread(img_file)
            #get initial centered image
            trans, centered = self.image_center(img)
            self.save_img(centered, outdir, name, success, trans)
            success +=1
            
            #trans, flip = self.flip(centered)
            #self.save_img(flip, outdir, name, success, trans)
            #success +=1
            
            #run transitions
            #for __ in range(3):
            #    degs = randint(5, 355)
            #    trans, rotated = self.rotate(img, degs)
            #    self.save_img(rotated, outdir, name, success, trans)
            #    success +=1 
        return success

    def check_complete(self, directory):
        jpgs = filter(lambda x: x.endswith('jpg'), os.listdir(directory))
        if len(jpgs)>0:
            return True
        return False
                
    def run(self):
        while True:
            job = self.queue.get()
            name = job['name']
            indir = job['indir']
            outdir = job['outdir']
            print "Starting {0}".format(name)
            success = 0
            start = time.time()
            valid_files, duplicate_counter, total_files= self.dedup(indir, outdir, name)               
            success = self.transform(valid_files, outdir, name)

            print """{0} finished. {1} successes, {2} duplicates of {3} files processed.
Comleted in {4:0.2f} seconds""".format(name, success, duplicate_counter, 
                                        total_files, time.time()-start)

                
            if self.queue.empty():
                break
    
            

    
        
