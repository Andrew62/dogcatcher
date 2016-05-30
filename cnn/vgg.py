


import tensorflow as tf
from cnn.vgg.util import kernel, bias, conv_layer, max_pool, matmul


class VGG(object):
    def __init__(self, n_classes=1000):

        layers = {
            1 : kernel([3, 3, 3, 64], 'conv1'),
            2 : kernel([3, 3, 64, 64], 'conv2'),

            3 : kernel([3, 3, 64, 128], 'conv3'),
            4 : kernel([3, 3, 128, 128], 'conv4'),

            5 : kernel([3, 3, 128, 256], 'conv5'),
            6 : kernel([3, 3, 256, 256], 'conv6'),
            7 : kernel([3, 3, 256, 256], 'conv7'),
            8 : kernel([3, 3, 256, 256], 'conv8'),

            9 : kernel([3, 3, 256, 512], 'conv9'),
            10 : kernel([3, 3, 512, 512], 'conv10'),
            11 : kernel([3, 3, 512, 512], 'conv11'),
            12 : kernel([3, 3, 512, 512], 'conv12'),

            #fc
            13 : kernel([4096, 4096], 'fc7'),
            14 : kernel([4096, n_classes], 'fc8')

        }

        biases = {
            1 : bias([64], 'bias1'),
            2 : bias([64], 'bias2'),

            3 : bias([128], 'bias3'),
            4 : bias([128], 'bias4'),

            5 : bias([256], 'bias5'),
            6 : bias([256], 'bias6'),
            7 : bias([256], 'bias7'),
            8 : bias([256], 'bias8'),

            9 : bias([512], 'bias9'),
            10 : bias([512], 'bias10'),
            11 : bias([512], 'bias11'),
            12 : bias([512], 'bias12'),

            13 : bias([4096], 'bias13'),
            14 : bias([n_classes], 'bias14')

        }

        # Don't need to specify input data shape once we know everything is hooked up
        # For reference, the verification shape is [256, 224, 224, 3]
        self.train_labels_placeholder = tf.placeholder(dtype=tf.float32, name="train_labels_placeholder", shape=[10, 224, 224, 3])
        self.train_data_placeholder = tf.placeholder(dtype=tf.float32, name="train_data_placeholder", shape=[10, 224, 224, 3])

        # TODO add validation and test inputs

        self.conv1 = conv_layer(self.train_data_placeholder, layers[1], biases[1], '1')
        self.conv2 = conv_layer(self.conv1, layers[2], biases[2], '2')
        self.pool1 = max_pool(self.conv2, 'pool1')

        self.conv3 = conv_layer(self.pool1, layers[3], biases[3], '3')
        self.conv4 = conv_layer(self.conv3, layers[4], biases[4], '4')
        self.pool2 = max_pool(self.conv4, 'pool2')

        self.conv5 = conv_layer(self.pool2, layers[5], biases[5], '5')
        self.conv6 = conv_layer(self.conv5, layers[6], biases[6], '6')
        self.conv7 = conv_layer(self.conv6, layers[7], biases[7], '7')
        self.conv8 = conv_layer(self.conv7, layers[8], biases[8], '8')
        self.pool3 = max_pool(self.conv8, 'pool3')

        self.conv9 = conv_layer(self.pool3, layers[9], biases[9], '9')
        self.conv10 = conv_layer(self.conv9, layers[10], biases[10], '10')
        self.conv11 = conv_layer(self.conv10, layers[11], biases[11], '11')
        self.conv12 = conv_layer(self.conv11, layers[12], biases[12], '12')
        self.pool3 = max_pool(self.conv12, 'pool4')

        self.fc6 = tf.reshape(self.pool3, [-1, 4096], 'fc6')
        self.fc7 = matmul(self.fc6, layers[13], biases[13], 'fc8')
        self.logits = matmul(self.fc7, layers[14], biases[14], 'logtis')

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.train_labels_placeholder))

        self.optimizer = tf.train.AdagradOptimizer(learning_rate=1e-3).minimize(self.loss)

        self.prediction = tf.nn.softmax(self.logits)



if __name__ == "__main__":
    vgg = VGG()
