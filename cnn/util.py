"""
General utilies for training. 
"""
import os
import csv
import time
import pickle
import numpy as np

def model_name(now):
    year = now.year
    month = now.month
    day = now.day
    return "model_{0}{1}{2}.ckpt".format(year, month, day)


def accuracy(predictions, labels):
    total = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    return total, (total * 1.) / predictions.shape[0]


def pkl_dump(obj, fp):
    with open(fp, 'wb') as target:
        pickle.dump(obj, target)

def pkl_load(fp):
    with open(fp, 'rb') as infile:
        return pickle.load(infile)

def write_csv(in_dict, fp):
    fields = ['iteration', 'loss', 'minibatch accuracy', 'valid accuracy']
    with open(fp, 'wb') as target:
        writer = csv.DictWriter(target, fields)
        writer.writeheader()
        for k, values in in_dict.items():
            values['iteration'] = k
            writer.writerow(values)


def get_last_checkpoint(model_dir):
    """
    USING tf implementation instead
    
    Loads the most recent checkpoint given a model dir. Returns None otherwise
    """
    files = os.listdir(model_dir)
    ckpts = filter(lambda x : x.endswith('ckpt'), files)
    most_recent_time = 0
    most_recent_ckpt = None
    for ckpt in ckpts:
        ckpt_path = os.path.join(model_dir, ckpt)
        stats = os.stat(ckpt_path)
        created_time = stats.st_ctime
        if created_time > most_recent_time:
            most_recent_time = created_time
            most_recent_ckpt = ckpt_path
    return most_recent_ckpt

def get_message(i, minibatch_accuracy, start, avg_loss, correct, epoch, learning_rate):
    subj = 'Iteration {0} Minibatch accuracy: {1:0.2%} ({2} correct)'.format(i + 1, minibatch_accuracy, correct)
    msg = "\n" + "*" * 50
    msg += '\nMinibatch loss at step {0}: {1:0.4f}\n'.format(i + 1, avg_loss)
    msg += subj + '\n'
    msg += "Epoch {0}\n".format(epoch)
    msg += 'Minibatch time: {0:0.0f} secs\n'.format(time.time() - start)
    msg += "Learning rate: {0}\n".format(learning_rate)
    msg += time.ctime()
    return subj, msg

