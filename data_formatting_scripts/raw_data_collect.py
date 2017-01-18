#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import pickle

top_dir = "/path/to/raw"

etna_counter = 0
data = []
for dir_, folders, files in os.walk(top_dir):
    for f in files:
        f_lower = f.lower()
        if f_lower.endswith(".jpg"):
            cls = os.path.basename(dir_)
            if 'Cirneco' in cls:
                etna_counter += 1
                cls = 'Etna Cirneco'
            data.append([cls, os.path.join(dir_, f)])

target_file = "/path/to/pickle.pkl"
with open(target_file, 'wb') as target:
    pickle.dump(data, target)

print("Done")
print(etna_counter)
