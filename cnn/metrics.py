import os
import csv
import util
import numpy as np
import tensorflow as tf
from nets import inception
from config import workspace
from datatf import read_one_image

# extras for showing imag

slim = tf.contrib.slim

BATCH_SIZE = 1
EPOCHS = 1

classes = util.pkl_load(workspace.class_pkl)
csv_files = workspace.test_csvs


graph = tf.Graph()
with graph.as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    fp_placeholder = tf.placeholder(tf.string)
    image = read_one_image(fp_placeholder)
    images = tf.reshape(image, [1, 224, 224, 3])

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(images, num_classes=len(classes), is_training=False)
    probabilities = tf.nn.softmax(logits)

    checkpoint = checkpoint_path = tf.train.latest_checkpoint(os.path.join(workspace.inception_cpkt, 'log'))
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint,
        slim.get_model_variables())

with tf.Session(graph=graph) as sess:
    init_fn(sess)
    top1 = 0
    top5 = 0
    total = 0
    # there is prob a better way to do this will all TF but
    # this will do for calculating accuracy.
    for csv_fp in workspace.test_csvs:
        with open(csv_fp) as infile:
            reader = csv.reader(infile)
            for line in reader:
                cls, fp = line
                cls = int(cls)
                feed_dict = {
                    fp_placeholder : fp
                }
                np_prob = sess.run(probabilities, feed_dict=feed_dict)
                np_prob = np_prob[0, 0:]

                # reverse the indexes
                pred_idxs = np.argsort(np_prob)[::-1]
                if pred_idxs[0] == cls:
                    top1 += 1
                if cls in pred_idxs[:5]:
                    top5 += 1
                total += 1

print("Top 1: {0:0.2%}\nTop 5: {1:0.2%}\n".format(top1/total, top5/total))
