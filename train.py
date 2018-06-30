from __future__ import division, print_function, absolute_import

import math
import random
import tensorflow as tf
import numpy as np
import os

import util.util as util

from data_sources.data_source import DataSource
from models.cnn import CNN

from learning.trainer import Trainer

import glob

def main():
    # Get dataset
    files = glob.glob('preprocessed_data/*.pickle')
    datasource = DataSource(files)

    # Get model
    learning_schedule=[
        {
            'loss_terms_to_optimize': {
                'heatmaps_mse': ['hourglass'],
                'radius_mse': ['radius'],
            },
            'learning_rate': 1e-3,
        },
    ]
    model = CNN(datasource.tensors, datasource.x_shape, learning_schedule)
    
    # Get trainer
    trainer = Trainer(model)

    # Train for 10000 steps
    return trainer.run_training(datasource, 10000)


if __name__ == '__main__':
    main()