from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import time
import pickle

class DataSource(object):
    def __init__(self,
                 train_files,
                 eval_files,
                 batch_size=32,
                 seed=7,
                 data_format='NHWC'):
        self.batch_size = batch_size
        self.data_format = data_format.upper()
        assert self.data_format == 'NHWC' or self.data_format == 'NCHW'

        self.train = Data(train_files, batch_size=batch_size, data_format=data_format)
        self.eval = Data(eval_files, batch_size=600, data_format=data_format)

        self.iter = tf.data.Iterator.from_structure(self.train._dataset.output_types,
                                                    self.train._dataset.output_shapes)
        
        self.train.make_initializer(self.iter)
        self.eval.make_initializer(self.iter)

        self.x_shape = (36, 60)
        self.tensors = self.iter.get_next()


class Data(object):
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
        base_dataset = base_dataset.map(self._set_shapes)
    
        self._dataset_single = base_dataset.cache().batch(self.batch_size)
        self._dataset = base_dataset.cache() \
                .shuffle(10000*self.batch_size, seed=seed) \
                .repeat().batch(self.batch_size) \
                .prefetch(2 * self.batch_size)
        
    def _preprocess_pickle(self, filename):
        data = pickle.load(open(filename, 'rb'))
        return data['eye'], data['heatmaps'], data['landmarks'], data['radius']
    
    def _set_shapes(self, eye, heatmaps, landmarks, radius):
        eye.set_shape([1, 36, 60])
        heatmaps.set_shape([18, 36, 60])
        landmarks.set_shape([18, 2])
        radius.set_shape([])
        return eye, heatmaps, landmarks, radius

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