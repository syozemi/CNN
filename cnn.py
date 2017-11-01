import tensorflow as tf
import numpy as np
import os
import pickle
import shutil
import cnn
from libs import (get_variable, get_conv, get_bias, get_pool, conv_and_pool)


class CNN:
    def __init__(self, input_sizex, input_sizey, num_class):
        with tf.Graph().as_default():
            self.input_sizex = input_sizex
            self.input_sizey = input_sizey
            self.num_class = num_class
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        input_sizex = self.input_sizex
        input_sizey = self.input_sizey
        num_class = self.num_class
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, input_sizex, input_sizey])
            x_image = tf.reshape(x, [-1,input_sizex,input_sizey,1])

        with tf.name_scope('conv_and_pool1'):
            num_filters1 = 16
            h_conv1, h_pool1 = conv_and_pool(x_image, 1, num_filters1, 10, 2)

        with tf.name_scope('conv_and_pool2'):
            num_filters2 = 32
            h_conv2, h_pool2 = conv_and_pool(h_pool1, num_filters1, num_filters2, 5, 1)

        with tf.name_scope('conv_and_pool3'):
            num_filters3 = 64
            h_conv3, h_pool3 = conv_and_pool(h_pool2, num_filters2, num_filters3, 3, 1)

        with tf.name_scope('conv_and_pool4'):
            num_filters4 = 128
            h_conv4, h_pool4 = conv_and_pool(h_pool3, num_filters3, num_filters4, 3, 1)

        with tf.name_scope('fully_connected'):
            final_pixel = 9
            h_pool_flat = tf.reshape(h_pool4, [-1, (final_pixel**2)*num_filters4])

            num_units1 = (final_pixel**2)*num_filters4
            num_units2 = 1024

            w2 = get_variable([num_units1, num_units2])
            b2 = get_bias([num_units2])
            hidden2 = tf.nn.relu(tf.matmul(h_pool_flat, w2) + b2)

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

        with tf.name_scope('softmax'):

            w0 = get_variable([num_units2, 1])
            b0 = get_bias([1])
            p_logits = tf.matmul(hidden2_drop, w0) + b0
            p = tf.sigmoid(p_logits)

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, 1])
            loss = tf.reduce_mean(tf.square(p - t))
            train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
            # correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # with tf.name_scope('evaluator'):
        #     correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", loss)
        # tf.summary.scalar("accuracy", accuracy)
        tf.summary.histogram("convolution_filters1", h_conv1)
        tf.summary.histogram("convolution_filters2", h_conv2)
        
        self.x, self.t, self.p, self.keep_prob = x, t, p, keep_prob
        self.train_step = train_step
        self.loss = loss
        self.p = p
        self.t = t

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()

        saver = tf.train.Saver()
        if os.path.isdir('/tmp/logs'):
            shutil.rmtree('/tmp/logs')
        writer = tf.summary.FileWriter("/tmp/logs", sess.graph)
        
        self.sess = sess
        self.summary = summary
        self.writer = writer
        self.saver = saver
