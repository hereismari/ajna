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
parser.add_argument('--input-image', type=str, required=True)
parser.add_argument('--model-checkpoint', type=str, required=True)


def main(args):
    # Get dataset
    datasource = DataSource(None, [args.input_image])

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
    
    # Get evaluator
    evaluator = Trainer(model)

    # Train for 10000 steps
    return evaluator.run_predict(datasource)


if __name__ == '__main__':
    main(parser.parse_args())