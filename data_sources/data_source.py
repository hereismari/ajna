from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import time
import pickle

class DataSource(object):
    def __init__(self,
                 files,
                 batch_size=32,
                 seed=7,
                 data_format='NHWC'):
        self.batch_size = batch_size

        self._files = files
        self._num_examples = len(files)
        self._ids = np.arange(self._num_examples)
        
        self.data_format = data_format.upper()
        assert self.data_format == 'NHWC' or self.data_format == 'NCHW'

        base_dataset = tf.data.Dataset.from_tensor_slices(self._files)

        base_dataset = base_dataset.map(lambda filename: tuple(tf.py_func(
            self._preprocess_pickle, [filename], [tf.float32, tf.float32, tf.float32, tf.float32])))

        self._dataset_single = base_dataset.cache().batch(self.batch_size)
        self._dataset = base_dataset.cache() \
                .shuffle(10000*self.batch_size, seed=seed) \
                .repeat().batch(self.batch_size) \
                .prefetch(2 * self.batch_size)

        self.iter = tf.data.Iterator.from_structure(self._dataset.output_types,
                                                    self._dataset.output_shapes)
        self.make_initializer(self.iter)
    
    def _preprocess_pickle(self, filename):
        data = pickle.load(open(filename, 'rb'))
        return data['eye'], data['heatmaps'], data['landmarks'], data['radius']

    def make_initializer(self, iter):
        self._init_op = iter.make_initializer(self._dataset)
        self._init_op_single = iter.make_initializer(self._dataset_single)
    
    def run(self, sess):
        sess.run(self._init_op)

    def run_single(self, sess):
        sess.run(self._init_op_single)

    @property
    def ids(self):
        return self._ids

    @property
    def num_examples(self):
        return self._num_examples