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
        self.placeholder_X = tf.placeholder(tf.float32, self.shape)
        # self.var_X = tf.Variable(self.placeholder_X)
        self.dataset = tf.data.Dataset.from_tensors((self.placeholder_X))
        self.iter = self.dataset.make_initializable_iterator()
        '''

        self.iter = tf.data.Iterator.from_structure(self.eval._dataset_single.output_types,
                                                    self.eval._dataset_single.output_shapes)
        
        if train_files is not None:
            self.train.make_initializer(self.iter)
        if eval_files is not None:
            self.eval.make_initializer(self.iter)
        '''
        self.x_shape = (36, 60)
        self.tensors = self.iter.get_next()

        self.eval = self

    def run_single(self, sess):
        sess.run(self.iter.initializer, feed_dict={self.placeholder_X: self.image.reshape(*self.shape)})