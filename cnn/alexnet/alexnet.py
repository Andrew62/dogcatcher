
"""
Tensorflow implementation of AlexNet
"""
import tensorflow as tf

class AlxNet(object):
    def __init__(self, n_classes, keep_prob=0.5, train=False, lrn=False):
        """
        The middle shape in AlexNet is designed to be 4096, however,
        we are using larger images so the fully connected layers
        need to be modified
        """
        self.input_data = tf.placeholder(tf.float32, name="input_data")#, shape=(128, 224, 224, 3))
        self.keep_prob = tf.constant(keep_prob, name="Dropout", dtype=tf.float32)
        self.train = train
        self.lrn = lrn

        self.n_classes = n_classes

        self.mean_subtract = tf.sub(self.input_data, tf.reduce_mean(self.input_data, reduction_indices=0))
        self.var_summary(self.mean_subtract, 'normed_images')

        with tf.variable_scope("pool1"):
            with tf.variable_scope('conv1'):
                self.weights1 = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32, stddev=1e-2))
                self.bias1 = tf.Variable(tf.constant(0.1, shape=[96], dtype=tf.float32))
                self.convolve1 = tf.nn.conv2d(self.mean_subtract, self.weights1, [1, 4, 4, 1], 'VALID')
                self.conv1 = tf.nn.relu(self.convolve1 + self.bias1)
                self.var_summary(self.conv1, 'conv1')

            if self.lrn is True:
                self.conv1 = tf.nn.local_response_normalization(self.conv1, depth_radius=2, alpha=2e-5,
                                                                beta=0.75, bias=1.0)
            self.pool1 = tf.nn.max_pool(self.conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")
            self.var_summary(self.pool1, 'pool1')


        with tf.variable_scope("pool2"):
            with tf.variable_scope('conv2'):
                self.weights2 = tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=1e-2, dtype=tf.float32))
                self.bias2 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[256]))
                self.convolve2 = tf.nn.conv2d(self.pool1, self.weights2, [1, 1, 1, 1], 'SAME')
                self.conv2 = tf.nn.relu(self.convolve2 + self.bias2)
                self.var_summary(self.conv2, 'conv2')

            if self.lrn is True:
                self.conv2 = tf.nn.local_response_normalization(self.conv2, depth_radius=2, alpha=2e-5,
                                                                beta=0.75, bias=1.0)
            self.pool2 = tf.nn.max_pool(self.conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")
            self.var_summary(self.pool2, 'pool2')

        with tf.variable_scope("pool3"):
            with tf.variable_scope('conv3'):
                self.weights3 = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32, stddev=1e-2))
                self.bias3 = tf.Variable(tf.constant(0.1, shape=[384], dtype=tf.float32))
                self.convolve3 = tf.nn.conv2d(self.pool2, self.weights3, [1, 1, 1, 1], 'SAME')
                self.conv3 = tf.nn.relu(self.convolve3 + self.bias3)
                self.var_summary(self.conv3, 'conv3')

            with tf.variable_scope("conv4"):
                self.weights4 = tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=1e-2, dtype=tf.float32))
                self.bias4 = tf.Variable(tf.constant(1.0, shape=[384], dtype=tf.float32))
                self.convolve4 = tf.nn.conv2d(self.conv3, self.weights4, [1, 1, 1, 1], 'SAME')
                self.conv4 = tf.nn.relu(self.convolve4 + self.bias4)
                self.var_summary(self.conv4, 'conv4')

            with tf.variable_scope("conv5"):
                self.weights5 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=1e-2, dtype=tf.float32))
                self.bias5 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[256]))
                self.convolve5 = tf.nn.conv2d(self.conv4, self.weights5, [1, 1, 1, 1], 'SAME')
                self.conv5 = tf.nn.relu(self.convolve5 + self.bias5)
                self.var_summary(self.conv5, 'conv5')

            self.pool5 = tf.nn.max_pool(self.conv5, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")
            self.var_summary(self.pool5, 'pool5')

            middle_shape = 6400

            self.reshape5 = tf.reshape(self.pool5, [-1, middle_shape])

        with tf.variable_scope("fc6"):
            self.weights6 = tf.Variable(tf.truncated_normal([middle_shape, 4096], dtype=tf.float32, stddev=1e-2))
            self.bias6 = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32))
            self.matmul_1 = tf.matmul(self.reshape5, self.weights6)
            self.fc6 = tf.nn.relu(self.matmul_1 + self.bias6)
            self.var_summary(self.fc6, 'fc6')

            if self.train is True:
                self.fc6 = tf.nn.dropout(self.fc6, self.keep_prob)

        with tf.variable_scope("fc7"):
            self.weights7 = tf.Variable(tf.truncated_normal([4096, 4096], stddev=1e-2, dtype=tf.float32))
            self.bias7 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[4096]))
            self.matmul_2 = tf.matmul(self.fc6, self.weights7)
            self.fc7 = tf.nn.relu(self.matmul_2 + self.bias7)
            self.var_summary(self.fc7, 'fc7')

            if self.train is True:
                self.fc7 = tf.nn.dropout(self.fc7, self.keep_prob)

        with tf.variable_scope("logits"):
            self.weights8 = tf.Variable(tf.truncated_normal([4096, self.n_classes], stddev=1e-2, dtype=tf.float32))
            self.bias8 = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[self.n_classes]))
            self.matmul_3 = tf.matmul(self.fc7, self.weights8)
            self.logits = tf.nn.relu(self.matmul_3 + self.bias8)
            self.var_summary(self.logits, 'logits')

        self.softmax = tf.nn.softmax(self.logits, 'softmax')

    @staticmethod
    def var_summary(var, name):
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.scalar_summary("mean/" + name, mean)
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary("stddev/" + name, stddev)
            tf.histogram_summary(name, var)

if __name__ == "__main__":
    alexnet = AlxNet(10)