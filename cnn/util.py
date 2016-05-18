
import pickle
import numpy as np

def model_name(now):
    year = now.year
    month = now.month
    day = now.day
    return "model_{0}{1}{2}.ckpt".format(year, month, day)


def accuracy(predictions, labels):
    return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) * 1.) / predictions.shape[0]


def pkl_dump(obj, fp):
    with open(fp, 'wb') as target:
        pickle.dump(obj, target)