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
from PIL import Image
from random import randint
from multiprocessing import Process
           

class MPTransformer(Process):

    def __init__(self, name, queue, pixels=224, transform=True):
        Process.__init__(self)
        self.queue = queue
        self.name = name
        self.pixels = pixels
        self.trans = transform
        
    def image_center(self, image):
        """
        reshapes, resizes, and returns a means subtracted image
        """
        return 'centered', image.resize((self.pixels, self.pixels))
        
    def rotate(self, image, degrees):
        return 'rotate_{0}'.format(degrees), image.rotate(degrees)
        
    def flip(self, image):
        return 'flipped', image.transpose(0)
    
    def save_hashes(self, outdir, name, img_hashes):
        with open(os.path.join(outdir, "{0}.pkl".format(name)), 'wb') as target:
            pickle.dump(img_hashes, target, pickle.HIGHEST_PROTOCOL)
    
    def load_hashes(self, indir):
        with open(indir, 'rb') as infile:
            return pickle.load(infile)
        
    def processed_hash(self, outdir, name):
        filep = os.path.join(outdir, "{0}.pkl".format(name))
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
                img_path = os.path.join(indir, img_file)
                img = Image.open(img_path)
                arr = np.array(img)
                if arr.shape[-1] != 3:
                    continue
                img_hash = hashlib.sha1(img.tobytes()).digest()
                if img_hash in already_used:
                    duplicate_counter += 1
                    continue
                already_used.add(img_hash)
                valid_files.append(img_path)
            except IOError:
                print "Couldn't open that one right"
            except ValueError:
                print "Nothing witty here"
            except IndexError:
                print os.path.basename(img_file)
                print "Not enough channles (Layers as Mary Barry would say)"
                
        self.save_hashes(outdir, name, already_used)
        return valid_files, duplicate_counter, len(img_files)
        
    def create_blank_array(self, valid_files):
        shape=(5*len(valid_files), self.pixels, self.pixels, 3)
        return np.zeros(shape=shape, dtype=np.uint8)
    
    def save_img(self, img, outdir, name, index, trans):
        name = "{0}_{1}_{2}.jpg".format(name, trans, index)
        path = os.path.join(outdir, name)
        img.save(path, 'JPEG')
        
    def transform(self, valid_files, outdir, name):
        success = 0
        for img_file in valid_files:
            img = Image.open(img_file)

            # get initial centered image
            trans, centered = self.image_center(img)
            self.save_img(centered, outdir, name, success, trans)
            success += 1

            if self.trans is True:
                trans, flip = self.flip(centered)
                self.save_img(flip, outdir, name, success, trans)
                success += 1

                for __ in range(3):
                    degs = randint(5, 355)
                    trans, rotated = self.rotate(img, degs)
                    self.save_img(rotated, outdir, name, success, trans)
                    success += 1
        return success

    def check_complete(self, directory):
        jpgs = filter(lambda x: x.endswith('jpg'), os.listdir(directory))
        if len(jpgs) > 0:
            return True
        return False

    def run(self):
        while True:
            job = self.queue.get()
            name = job['name']
            indir = job['indir']
            outdir = job['outdir']
            print "Starting {0}".format(name)
            start = time.time()
            valid_files, duplicate_counter, total_files = self.dedup(indir, outdir, name)
            success = self.transform(valid_files, outdir, name)

            print """{0} finished. {1} successes, {2} duplicates of {3} files processed.
Comleted in {4:0.2f} seconds""".format(name, success, duplicate_counter,
                                       total_files, time.time()-start)

            if self.queue.empty():
                break

if __name__ == "__main__":
    from Queue import Queue
    q = Queue()
    job = dict(name="Affenpinscher", indir="/home/andrew/Documents/dogcatcher/raw_data/Affenpinscher",
               outdir="/home/andrew/Downloads")
    q.put(job)
    process = MPTransformer('1', q)
    process.run()
    