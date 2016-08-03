
"""
Tensorflow implementation of AlexNet
"""

import tensorflow as tf

class AlxNet(object):
    def __init__(self, n_classes, keep_prob=0.5, train=False):
        """
        The middle shape in AlexNet is designed to be 4096, however,
        we are using larger images so the fully connected layers
        need to be modified
        """
        self.input_data = tf.placeholder(dtype=tf.float32, name="input_data")
        self.keep_prob = tf.constant(keep_prob, name="Dropout", dtype=tf.float32)

        # Used to convert the final conv layer to a dense layer
        self.middle_shape = 43264

        mean = tf.constant([125.974950491, 121.990847064, 102.991749558],
                           dtype=tf.float32, name='img_mean')

        self.mean_subtract = self.input_data - mean

        with tf.variable_scope("pool1"):
            with tf.variable_scope('conv1'):
                self.weights1 = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32, stddev=1e-2))
                self.bias1 = tf.Variable(tf.constant(1.0, shape=[96], dtype=tf.float32))
                self.conv1 = tf.nn.conv2d(self.mean_subtract, self.weights1, [1, 4, 4, 1], 'SAME')
                self.hidden1 = tf.nn.relu(self.conv1 + self.bias1)
                self.response_norm1 = tf.nn.local_response_normalization(self.hidden1, depth_radius=5, alpha=1e-3,
                                                                    beta=0.75, bias=2.0)
            self.pool1 = tf.nn.max_pool(self.response_norm1, [1, 2, 2, 1], [1, 1, 1, 1], padding="VALID")

        with tf.variable_scope("pool2"):
            with tf.variable_scope('conv2'):
                self.weights2 = tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=1e-2, dtype=tf.float32))
                self.bias2 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[256]))
                self.conv2 = tf.nn.conv2d(self.pool1, self.weights2, [1, 1, 1, 1], 'SAME')
                self.hidden2 = tf.nn.relu(self.conv2 + self.bias2)
                self.response_norm2 = tf.nn.local_response_normalization(self.hidden2, depth_radius=5, alpha=1e-3,
                                                                    beta=0.75, bias=2.0)
            self.pool2 = tf.nn.max_pool(self.response_norm2, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

        with tf.variable_scope("pool3"):
            with tf.variable_scope('conv3'):
                self.weights3 = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32, stddev=1e-2))
                self.bias3 = tf.Variable(tf.constant(1.0, shape=[384], dtype=tf.float32))
                self.conv3 = tf.nn.conv2d(self.pool2, self.weights3, [1, 2, 2, 1], 'VALID')
                self.hidden3 = tf.nn.relu(self.conv3 + self.bias3)

            with tf.variable_scope("conv4"):
                self.weights4 = tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=1e-2, dtype=tf.float32))
                self.bias4 = tf.Variable(tf.constant(1.0, shape=[384], dtype=tf.float32))
                self.conv4 = tf.nn.conv2d(self.hidden3, self.weights4, [1, 1, 1, 1], 'SAME')
                self.hidden4 = tf.nn.relu(self.conv4 + self.bias4)

            with tf.variable_scope("conv5"):
                self.weights5 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=1e-2, dtype=tf.float32))
                self.bias5 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[256]))
                self.conv5 = tf.nn.conv2d(self.hidden4, self.weights5, [1, 1, 1, 1], 'SAME')
                self.hidden5 = tf.nn.relu(self.conv5 + self.bias5)

            self.pool5 = tf.nn.max_pool(self.hidden5, [1, 2, 2, 1], [1, 1, 1, 1], padding="SAME")
            self.reshape5 = tf.reshape(self.pool5, [-1, self.middle_shape])

        with tf.variable_scope("fc6"):
            self.weights6 = tf.Variable(tf.truncated_normal([self.middle_shape, 4096], dtype=tf.float32, stddev=1e-2))
            self.bias6 = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32))
            self.matmul_1 = tf.matmul(self.reshape5, self.weights6)
            self.fc6 = tf.nn.relu(self.matmul_1 + self.bias6)

            if train is True:
                self.fc6 = tf.nn.dropout(self.fc6, keep_prob)

        with tf.variable_scope("fc7"):
            self.weights7 = tf.Variable(tf.truncated_normal([4096, 4096], stddev=1e-2, dtype=tf.float32))
            self.bias7 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[4096]))
            self.matmul_2 = tf.matmul(self.fc6, self.weights7)
            self.fc7 = tf.nn.relu(self.matmul_2 + self.bias7)

            if train is True:
                self.fc7 = tf.nn.dropout(self.fc7, keep_prob)

        with tf.variable_scope("logits"):
            self.weights8 = tf.Variable(tf.truncated_normal([4096, n_classes], stddev=1e-2, dtype=tf.float32))
            self.bias8 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[n_classes]))
            self.logits = tf.nn.bias_add(tf.matmul(self.fc7, self.weights8), self.bias8)

        self.softmax = tf.nn.softmax(self.logits, 'softmax')


if __name__ == "__main__":
    alexnet = AlxNet(10)