# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 16:03:40 2016

@author: Andrew
github: Andrew62

"""

import os
import time
import shutil
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from data import DataSet
from config import workspace
from datetime import datetime

#need tensorflow_serving to run model
#intall http://tensorflow.github.io/serving/setup
#see http://tensorflow.github.io/serving/serving_basic
#need to download bazel
#https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md
#from tensorflow_serving.session_bundle import exporter

              
def model_name(now):
    year = now.year
    month = now.month
    day = now.day
    return "model_{0}{1}{2}.ckpt".format(year, month, day)
    
def accuracy(predictions, labels):
    return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))*1.)/predictions.shape[0]

def pkl_dump(obj, fp):
    with open(fp, 'wb') as target:
        pickle.dump(obj, target)

data = DataSet(workspace.train_pkl, workspace.test_pkl, 
              workspace.valid_pkl, workspace.class_pkl,
              img_shape=(256,256,3))  
NUM_CORES=4
MESSAGE_EVERY = 100
BATCH_SIZE = 256
N_CLASSES = data.n_classes
train_dat_size = (BATCH_SIZE, 256, 256, 3)
train_lab_size = (BATCH_SIZE, 252)


test_data, test_labels = data.test_batch(BATCH_SIZE)
valid_data, valid_labels = data.valid_batch(BATCH_SIZE)

graph = tf.Graph()
with graph.as_default():
    
    tf_test_data = tf.constant(test_data, name="tf_test_data", 
                               dtype=tf.float32, shape=train_dat_size)
                               
    keep_prob = tf.constant(0.5, name="dropout", dtype=tf.float32)
    
    valid_data_constant = tf.constant(valid_data, name="valid_data_constant", 
                                      dtype=tf.float32, shape=train_dat_size)

    train_data_placeholder = tf.placeholder(dtype=tf.float32,
                                            shape=train_dat_size,
                                            name="train_data_placeholder")
    train_labels_placeholder = tf.placeholder(dtype=tf.float32,
                                              shape = train_lab_size,
                                              name='train_labels_placeholder')

    kernel_1 = tf.Variable(tf.truncated_normal([11,11,3,48], dtype=tf.float32,
                                               stddev=1e-2, name='kernel_1'))
    biases_1 = tf.Variable(tf.constant(0, shape=[48], dtype=tf.float32, 
                                       name='biases_1'))
    kernel_2 = tf.Variable(tf.truncated_normal([5,5,48,128], dtype=tf.float32,
                                               stddev=1e-2, name='kernel_2'))
    biases_2 = tf.Variable(tf.constant(1.0, shape=[128], dtype=tf.float32, 
                                       name='biases_2'))
                                                                    
    kernel_3 = tf.Variable(tf.truncated_normal([3, 3, 128, 192], dtype=tf.float32,
                                               stddev=1e-2, name='kernel_3'))
    biases_3 = tf.Variable(tf.constant(0, shape=[192], dtype=tf.float32,
                                       name='biases_3'))
    kernel_4 = tf.Variable(tf.truncated_normal([3 ,3, 192, 192], dtype=tf.float32,
                                               stddev=1e-2, name='kernel_4'))
    biases_4 = tf.Variable(tf.constant(1.0, shape=[192], dtype=tf.float32,
                                       name='biases_4'))
    kernel_5 = tf.Variable(tf.truncated_normal([3, 3, 192, 128], dtype=tf.float32, 
                                               stddev=1e-1, name='kernel_5'))
    biases_5 = tf.Variable(tf.constant(1.0, shape=[128], dtype=tf.float32,
                                       name='biases_5')) 
    #middle_shape = pool_5_shape[1]*pool_5_shape[2] * pool_5_shape[3]
    middle_shape = 8192
                          
    layer_6 = tf.Variable(tf.truncated_normal([middle_shape, middle_shape], dtype=tf.float32, 
                                              stddev=1e-2, name='layer_6'))                                  
    biases_6 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[middle_shape],
                                       name='biases_6'))
    layer_7 = tf.Variable(tf.truncated_normal([middle_shape, middle_shape], dtype=tf.float32,
                                              stddev=1e-2, name='layer_7'))
    biases_7 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[middle_shape],
                                       name='biases_7'))
    layer_8 = tf.Variable(tf.truncated_normal([middle_shape, N_CLASSES], dtype=tf.float32,
                                              stddev=1e-2, name='layer_8'))
    biases_8 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[N_CLASSES],
                                       name="biases_8"))                                              
    
    def model(data, train=False):             
        conv_1 = tf.nn.conv2d(data, kernel_1, [1,4,4,1], 
                              padding='SAME', name='convolution_1')
    
        hidden_1 = tf.nn.relu(conv_1+biases_1, name='ReLU_1')
        
        norm_1 = tf.nn.local_response_normalization(hidden_1, 
                                                    depth_radius = 2,
                                                    alpha = 2e-05, 
                                                    beta = 0.75, 
                                                    bias = 1.0,
                                                    name='norm_1')
        
        max_pool_1 = tf.nn.max_pool(norm_1, [1,3,3,1],[1,2,2,1], "SAME", 
                                    name='max_pool_1')
                                    
        conv_2 = tf.nn.conv2d(max_pool_1, kernel_2, strides=[1,1,1,1],
                                 padding='SAME', name='conv_2')
        hidden_2 = tf.nn.relu(conv_2+biases_2, name='ReLU_2')
        
        norm_2 = tf.nn.local_response_normalization(hidden_2,
                                                    depth_radius = 2,
                                                    alpha = 2e-05, 
                                                    beta = 0.75, 
                                                    bias = 1.0,
                                                    name='norm_2')
        
        max_pool_2 = tf.nn.max_pool(norm_2, [1,3,3,1],[1,2,2,1], "SAME", 
                                    name='max_pool_2')
       
        conv_3 = tf.nn.conv2d(max_pool_2, kernel_3, strides=[1,1,1,1],
                              padding='SAME', name='conv_3')
        
        hidden_3 = tf.nn.relu(conv_3+biases_3, name='relu_3')
    
        conv_4 = tf.nn.conv2d(hidden_3, kernel_4, strides=[1,1,1,1],
                              padding='SAME', name='conv_4')
        hidden_4 = tf.nn.relu(conv_4+biases_4, name='hidden_4')
        
        conv_5 = tf.nn.conv2d(hidden_4, kernel_5, strides=[1,1,1,1], 
                              padding='SAME', name='conv_5')
        max_pool_5 = tf.nn.max_pool(conv_5, [1,3,3,1], [1,2,2,1], 
                                    padding='SAME',name='max_pool_5')  
#        pool_5_shape = max_pool_5.get_shape().as_list()[0]
        
        reshape_max_pool_5 = tf.reshape(max_pool_5,[-1, middle_shape])
        
        matmul_6 = tf.matmul(reshape_max_pool_5, layer_6)
        
        hidden_6 = tf.nn.relu(matmul_6+biases_6, name='hidden_6')
        
        if train is True:
            hidden_6 = tf.nn.dropout(hidden_6, keep_prob)
        
        matmul_7 = tf.matmul(hidden_6, layer_7)
        
        hidden_7 = tf.nn.relu(matmul_7+biases_7, name='hidden_7')
        
        if train is True:
            hidden_7 = tf.nn.dropout(hidden_7, keep_prob)
        
        matmul_8 = tf.matmul(hidden_7, layer_8)
        
        return tf.nn.relu(matmul_8+biases_8, name="logits")
        
    logits = model(train_data_placeholder, train=True)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_placeholder))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    

    
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(valid_data_constant))
    test_prediction = tf.nn.softmax(model(tf_test_data))
    


config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                        intra_op_parallelism_threads=NUM_CORES)
                        
with tf.device("/cpu:0"):
    with tf.Session(graph=graph, config=config) as sess:
        tf.initialize_all_variables().run()
        print "\n","*"*50
        print "Batch size: {0} images".format(BATCH_SIZE)
        print "Initialized"
        saver = tf.train.Saver(tf.all_variables(), name='dogcatcher', 
                               keep_checkpoint_every_n_hours=24)
        performance_data = {}
        try:
            for i in xrange(50001):
                performance_data[i]={}
                start = time.time()
                train_dat, train_lab = data.train_batch(BATCH_SIZE)
                feed_dict = {train_data_placeholder: train_dat,
                             train_labels_placeholder: train_lab}
                _, sess_loss, predictions = sess.run([optimizer, loss, train_prediction], 
                                                     feed_dict=feed_dict)
                
                
                if (i+1) % MESSAGE_EVERY == 0:
                    minibatch_accuracy = accuracy(predictions, train_lab)   
                    valid_accuracy = accuracy(valid_prediction.eval(), valid_labels)
                    
                    #collecting data for visualization later. Could prob use 
                    #tensorboard                                 
                    performance_data[i]['loss']=sess_loss.mean()
                    performance_data[i]['minibatch accuracy']=minibatch_accuracy
                    performance_data[i]['valid accuracy'] = valid_accuracy
                    
                    print "\n","*"*50
                    print 'Minibatch loss at step {0}: {1:0.6f}'.format(i+1, sess_loss.mean())
                    print 'Minibatch accuracy: {0:0.2%}'.format(minibatch_accuracy)
                    print "Valid accuracy: {0:0.2%}".format(valid_accuracy)
                    print 'Minibatch time: {0:0.0f} secs'.format(time.time() - start)
                    print time.ctime()
                if (i+1) % 5000:
                    saver.save(sess, os.path.join(workspace.model_dir, model_name(datetime.now())))
            print "\n","*"*50
            print "\n","*"*50
            print 'Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels)
        except Exception as e:
            print e
            print "Failed after {0} steps".format(i)
        
        finally:
            saver.save(sess, os.path.join(workspace.model_dir, model_name(datetime.now())))
            outg = os.path.join(workspace.model_dir, "graph")
            if os.path.exists(outg):
                shutil.rmtree(outg)
            tf.train.write_graph(sess.graph_def, outg, "graph.pb")
            performance_frame = pd.DataFrame.from_dict(performance_data, orient='index')
            performance_frame.to_csv(os.path.join(workspace.model_dir, 'performance.csv'), 
                                     index=False)
            #Probably want to separate this encoder out from the data class
            onehot = data.encoder
            pkl_dump(onehot, os.path.join(workspace.model_dir, "encoder.py"))
    
