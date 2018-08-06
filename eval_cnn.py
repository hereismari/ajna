from __future__ import division, print_function, absolute_import

import argparse
import os
import util.util as util

from data_sources.data_source import DataSource
from models.cnn import CNN
from learning.trainer import Trainer
import glob

parser = argparse.ArgumentParser(description='Eval CNN (Elg).')
parser.add_argument('--test-path', type=str, required=True)
parser.add_argument('--model-checkpoint', type=str, required=True)

parser.add_argument('--eye-shape', type=int, nargs="+", default=[90, 60])
parser.add_argument('--heatmap-scale', type=float, default=1)
parser.add_argument('--data-format', type=str, default='NCHW')


def main(args):
    # Get dataset
    test_files = glob.glob(os.path.join(args.test_path, '*.pickle'))
    datasource = DataSource(None, test_files, shape=tuple(args.eye_shape),
                            data_format=args.data_format, heatmap_scale=args.heatmap_scale)

    # Get model
    learning_schedule=[
        {
            'loss_terms_to_optimize': {
                'heatmaps_mse': ['hourglass'],
                'radius_mse': ['radius'],
            },
            'learning_rate': 1e-3,
        }
    ]
    model = CNN(datasource.tensors, datasource.x_shape, learning_schedule)
    
    # Get evaluator
    evaluator = Trainer(model, model_checkpoint=args.model_checkpoint)

    # Evaluate
    avg_losses = evaluator.run_eval(datasource)
    print('Avarage Losses', avg_losses)


if __name__ == '__main__':
    main(parser.parse_args())