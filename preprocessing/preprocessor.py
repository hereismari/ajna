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
    def __init__(self, data_format='NHWC', output_path='pretrained_path/', input_path='datasets'):
        """Initialize a data source instance."""
        # assert tensorflow_session is not None and isinstance(tensorflow_session, tf.Session)
        self.data_format = data_format.upper()
        self._output_path = output_path
        self._input_path = input_path
        assert self.data_format == 'NHWC' or self.data_format == 'NCHW'

    def preprocess_entry(self, entry):
        """Preprocess a "raw" data entry."""
        pass
    
    def preprocess_data(self, data):
        """Preprocess inputs from self._input_path and saves at self._output_path."""
        pass