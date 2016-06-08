#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""

import util
import tensorflow as tf
from config import workspace

def input_pipline(fp, encoder, batch_size=256, epochs=4):
    """
    Reads input csv and returns an input queue
    """
    paths = []
    labels = []
    for label, path in util.pkl_load(fp):
        paths.append(path)
        labels.append(label)
    labels = encoder.encode(labels)
    paths_tf = tf.convert_to_tensor(paths, dtype=tf.string, name="image_paths")
    labels_tf = tf.convert_to_tensor(labels, dtype=tf.float32, name="encoded_labels")
    producer = tf.train.slice_input_producer([paths_tf, labels_tf], name="producer", num_epochs=epochs,
                                             capacity=(3*batch_size))
    img_float, encoded_label = read_image(producer)
    return tf.train.batch([img_float, encoded_label], batch_size=batch_size, capacity=(3*batch_size), name="batcher", num_threads=2)


def read_image(queue):
    paths, labels = queue
    file_contents = tf.read_file(paths)
    imgs = tf.image.decode_jpeg(file_contents, channels=3, name="input_jpg")
    cast = tf.cast(imgs, dtype=tf.float32, name="float_img")
    cast.set_shape(list(workspace.img_shape))
    return cast, labels


if __name__ == "__main__":
    print "running"
    from cnn.config import workspace
    from cnn.encoder import  OneHot
    from hashlib import md5
    print "loading pkl"
    cls = util.pkl_load(workspace.class_pkl)
    print "creating encoder"
    encoder = OneHot(cls)
    print "creating graph"
    with tf.Graph().as_default():
        img, label = input_pipline(workspace.test_pkl, encoder, 100, 1)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                i = 0
                while not coord.should_stop():
                    old = None
                    out_label = sess.run([label])
            except tf.errors.OutOfRangeError:
                print "reached epoch limit"
            finally:
                coord.request_stop()
            coord.join(threads)
    print 'done'
