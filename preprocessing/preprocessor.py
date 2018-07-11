"""
Originally from: https://github.com/swook/GazeML
Modified by: @mari-linhares

Default specification of a data source.
"""

from collections import OrderedDict
import multiprocessing
import queue
import threading
import time

import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


class Preprocessor(object):
    """Base Preprocessor class."""
    def __init__(self, data_format='NHWC'):
        """Initialize a data source instance."""
        # assert tensorflow_session is not None and isinstance(tensorflow_session, tf.Session)
        self.data_format = data_format.upper()
        assert self.data_format == 'NHWC' or self.data_format == 'NCHW'

    def _determine_dtypes_and_shapes(self):
        """Determine the dtypes and shapes of Tensorflow queue and staging area entries."""
        while True:
            raw_entry = next(self.entry_generator(yield_just_one=True))
            if raw_entry is None:
                continue
            preprocessed_entry_dict = self.preprocess_entry(raw_entry)
            if preprocessed_entry_dict is not None:
                break
        labels, values = zip(*list(preprocessed_entry_dict.items()))
        dtypes = [value.dtype for value in values]
        shapes = [value.shape for value in values]
        return labels, dtypes, shapes

    def preprocess_entry(self, entry):
        """Preprocess a "raw" data entry and saves in preprocessed_data/."""
        pass