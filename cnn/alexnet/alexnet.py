
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
        self.input_data = tf.placeholder(tf.float32, name="input_data")#, shape=(256, 224, 224, 3))
        self.keep_prob = tf.constant(keep_prob, name="Dropout", dtype=tf.float32)
        self.train = train

        self.n_classes = n_classes

        with tf.variable_scope("pool1"):
            with tf.variable_scope('conv1'):
                self.weights1 = tf.get_variable('weights', [11, 11, 3, 48],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias1 = tf.get_variable('bias', [48], initializer=tf.constant_initializer(0.0))
                self.convolve1 = tf.nn.conv2d(self.input_data, self.weights1, [1, 4, 4, 1], 'SAME')
                self.conv1 = tf.nn.relu(self.convolve1 + self.bias1)
            self.norm1 = tf.nn.local_response_normalization(self.conv1, depth_radius=2, alpha=2e-5,
                                                            beta=0.75, bias=1.0)
            self.pool1 = tf.nn.max_pool(self.norm1, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")


        with tf.variable_scope("pool2"):
            with tf.variable_scope('conv2'):
                self.weights2 = tf.get_variable('weights', [5, 5, 48, 128],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias2 = tf.get_variable('bias', [128], initializer=tf.constant_initializer(1.0))
                self.convolve2 = tf.nn.conv2d(self.pool1, self.weights2, [1, 1, 1, 1], 'SAME')
                self.conv2 = tf.nn.relu(self.convolve2 + self.bias2)
            self.norm2 = tf.nn.local_response_normalization(self.conv2, depth_radius=2, alpha=2e-5,
                                                            beta=0.75, bias=1.0)
            self.pool2 = tf.nn.max_pool(self.norm2, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")

        with tf.variable_scope("pool3"):
            with tf.variable_scope('conv3'):
                self.weights3 = tf.get_variable('weights', [3, 3, 128, 192],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias3 = tf.get_variable('bias', [192], initializer=tf.constant_initializer(0.0))
                self.convolve3 = tf.nn.conv2d(self.pool2, self.weights3, [1, 1, 1, 1], 'SAME')
                self.conv3 = tf.nn.relu(self.convolve3 + self.bias3)

            with tf.variable_scope("conv4"):
                self.weights4 = tf.get_variable('weights', [3, 3, 192, 192],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias4 = tf.get_variable('bias', [192], initializer=tf.constant_initializer(1.0))
                self.convolve4 = tf.nn.conv2d(self.conv3, self.weights4, [1, 1, 1, 1], 'SAME')
                self.conv4 = tf.nn.relu(self.convolve4 + self.bias4)

            with tf.variable_scope("conv5"):
                self.weights5 = tf.get_variable('weights', [3, 3, 192, 128],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias5 = tf.get_variable('bias', [128], initializer=tf.constant_initializer(1.0))
                self.convolve5 = tf.nn.conv2d(self.conv4, self.weights5, [1, 1, 1, 1], 'SAME')
                self.conv5 = tf.nn.relu(self.convolve5 + self.bias5)

            self.pool5 = tf.nn.max_pool(self.conv5, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")
            # pool_5_shape = self.pool5.get_shape().as_list()[1:]
            # middle_shape = 1
            # for x in pool_5_shape:
            #     middle_shape *= x
            middle_shape = 6272

            self.reshape5 = tf.reshape(self.pool5, [-1, middle_shape])

        with tf.variable_scope("fc6"):
            self.weights6 = tf.get_variable('weights', [middle_shape, 4096],
                            initializer=tf.random_normal_initializer(stddev=1e-2))
            self.bias6 = tf.get_variable('bias', [4096], initializer=tf.constant_initializer(1.0))
            self.matmul_1 = tf.matmul(self.reshape5, self.weights6)
            self.fc6 = tf.nn.relu(self.matmul_1 + self.bias6)
            if self.train is True:
                self.fc6 = tf.nn.dropout(self.fc6, self.keep_prob)

        with tf.variable_scope("fc7"):
            self.weights7 = tf.get_variable('weights', [4096, 4096],
                                            initializer=tf.random_normal_initializer(stddev=1e-2))
            self.bias7 = tf.get_variable('bias', [4096], initializer=tf.constant_initializer(1.0))
            self.matmul_2 = tf.matmul(self.fc6, self.weights7)
            self.fc7 = tf.nn.relu(self.matmul_2 + self.bias7)
            if self.train is True:
                self.fc7 = tf.nn.dropout(self.fc7, self.keep_prob)

        with tf.variable_scope("logits"):
            self.weights8 = tf.get_variable('weights', [4096, self.n_classes],
                                            initializer=tf.random_normal_initializer(stddev=1e-2))
            self.bias8 = tf.get_variable('bias', [self.n_classes], initializer=tf.constant_initializer(1.0))
            self.matmul_3 = tf.matmul(self.fc7, self.weights8)
            self.logits = tf.nn.relu(self.matmul_3)

        self.softmax = tf.nn.softmax(self.logits, 'softmax')

if __name__ == "__main__":
    alexnet = AlxNet(10)