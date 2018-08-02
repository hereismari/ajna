from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import time
import pickle

class DataSource(object):
    def __init__(self,
                 train_files=None,
                 eval_files=None,
                 batch_size=32,
                 seed=7,
                 shape=(150, 90),
                 heatmap_scale=0.5,
                 data_format='NHWC'):
        self.batch_size = batch_size
        self.data_format = data_format.upper()
        assert self.data_format == 'NHWC' or self.data_format == 'NCHW'

        if train_files is not None:
            self.train = Data(train_files, batch_size=batch_size, data_format=data_format,
                              heatmap_scale=heatmap_scale, shape=shape)
       
        if eval_files is not None:
            self.eval = Data(eval_files, batch_size=batch_size, data_format=data_format,
                             heatmap_scale=heatmap_scale, shape=shape)

        self.iter = tf.data.Iterator.from_structure(self.eval._dataset.output_types,
                                                    self.eval._dataset.output_shapes)
        
        if train_files is not None:
            self.train.make_initializer(self.iter)
        if eval_files is not None:
            self.eval.make_initializer(self.iter)

        self.x_shape = (36, 60)
        self.tensors = self.iter.get_next()


class Data(object):
    def __init__(self,
                 files,
                 batch_size=32,
                 shape=(150, 90),
                 heatmap_scale=0.5,
                 seed=7,
                 data_format='NHWC'):
        self.batch_size = batch_size

        self._files = files
        self._num_examples = len(files)
        self._ids = np.arange(self._num_examples)
        
        self.data_format = data_format.upper()
        assert self.data_format == 'NHWC' or self.data_format == 'NCHW'

        self._heatmap_scale = heatmap_scale
        self._shape = shape

        base_dataset = tf.data.Dataset.from_tensor_slices(self._files)

        base_dataset = base_dataset.map(lambda filename: tuple(tf.py_func(
            self._preprocess_pickle, [filename], [tf.float32, tf.float32, tf.float32, tf.float32])))
        base_dataset = base_dataset.map(self._set_shapes)
    
        self._dataset_single = base_dataset.cache().batch(self.batch_size)
        self._dataset = base_dataset.cache() \
                .shuffle(self._num_examples + 100, seed=seed) \
                .repeat().batch(self.batch_size) \
                .prefetch(2 * self.batch_size)
        
    def _preprocess_pickle(self, filename):
        data = pickle.load(open(filename, 'rb'))
        return data['eye'], data['heatmaps'], data['landmarks'], data['radius']
    
    def _set_shapes(self, eye, heatmaps, landmarks, radius):
        heatmaps_shape = [int(s * self._heatmap_scale) for s in self._shape]
        if self.data_format == 'NHWC':
            eye.set_shape(list(self._shape) + [1])
            heatmaps.set_shape(heatmaps_shape + [18])
        else:
            eye.set_shape([1] + list(self._shape))
            heatmaps.set_shape([18] + heatmaps_shape)
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