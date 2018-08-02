from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

class ImgDataSource(object):
    def __init__(self,
                 shape=(150, 90),
                 data_format='NHWC'):
        
        self.image = None
        self.data_format = data_format.upper()
        assert self.data_format == 'NHWC' or self.data_format == 'NCHW'

        self.shape = (1, shape[0], shape[1], 1) if self.data_format == 'NHWC' else (1, 1, shape[0], shape[1])
        self.x_shape = shape
        
        self.placeholder_X = tf.placeholder(tf.float32, self.shape)
        self.dataset = tf.data.Dataset.from_tensors((self.placeholder_X))
        self.iter = self.dataset.make_initializable_iterator()
        
        self.tensors = self.iter.get_next()

        self.eval = self

    def run_single(self, sess):
        sess.run(self.iter.initializer, feed_dict={self.placeholder_X: self.image.reshape(*self.shape)})