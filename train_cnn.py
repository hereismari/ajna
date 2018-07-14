from __future__ import division, print_function, absolute_import

import os
import glob
import util.util as util

from data_sources.data_source import DataSource
from models.cnn import CNN
from learning.trainer import Trainer


import argparse
parser = argparse.ArgumentParser(description='Train CNN (Elg).')
parser.add_argument('--train-path', type=str, default='data/', required=True)
parser.add_argument('--eval-path', type=str, default='data/', required=True)
parser.add_argument('--steps', type=int, default=2000)

parser.add_argument('--eye-shape', type=int, nargs="+", default=[90, 60])
parser.add_argument('--heatmap-scale', type=float, default=1)
parser.add_argument('--data-format', type=str, default='NCHW')


def main(args):
    # Get dataset
    train_files = glob.glob(os.path.join(args.train_path, '*.pickle'))
    eval_files = glob.glob(os.path.join(args.eval_path, '*.pickle'))
    datasource = DataSource(train_files, eval_files, shape=tuple(args.eye_shape),
                            data_format=args.data_format, heatmap_scale=args.heatmap_scale)

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
    return trainer.run_training(datasource, args.steps)


if __name__ == '__main__':
    main(parser.parse_args())