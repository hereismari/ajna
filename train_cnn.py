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

import argparse
parser = argparse.ArgumentParser(description='Train CNN (Elg).')
parser.add_argument('--train-path', type=str, default='data/', required=True)
parser.add_argument('--eval-path', type=str, default='data/', required=True)

def main(args):
    # Get dataset
    train_files = glob.glob(os.path.join(args.train_path, '*.pickle'))
    eval_files = glob.glob(os.path.join(args.eval_path, '*.pickle'))
    datasource = DataSource(train_files, eval_files)

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
    return trainer.run_training(datasource, 1000)


if __name__ == '__main__':
    main(parser.parse_args())