"""
Tensorflow wrapper implemented independent of the alexnet wrapper.
Any VGG specific support code will go in the VGG folder
"""

import numpy as np
import pickle as pkl

def accuracy(predictions, labels):
    return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) * 1.) / predictions.shape[0]


def pkl_dump(obj, fp):
    with open(fp, 'wb') as target:
        pkl.dump(obj, target)

def pkl_load(fp):
    with open(fp, 'rb') as infile:
        return pkl.load(infile)

def get_middle_shape(tensor):
    """
    Used to find that intermediate shape for fully connected layers
    :param tensor:
    :return: an integer
    """
    return reduce(lambda x, y : x * y, tensor.get_shape().as_list()[1:])

def __make_cls_pkl__():
    import csv
    from config import workspace
    classes = set()
    with open(workspace.data_csv, 'rb') as infile:
        reader = csv.reader(infile)
        for path, cls in reader:
            classes.add(cls)
    classes = list(classes)
    pkl_dump(classes, workspace.class_pkl)

if __name__ == "__main__":
    __make_cls_pkl__()