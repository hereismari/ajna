from __future__ import division, print_function, absolute_import


import util.util as util

from data_sources.data_source import DataSource
from models.cnn import CNN
from learning.trainer import Trainer

import argparse

parser = argparse.ArgumentParser(description='Train CNN (Elg).')
parser.add_argument('--input-image', type=str, required=True)
parser.add_argument('--model-checkpoint', type=str, default='checkpoints/best_cnn.ckpt')

parser.add_argument('--eye-shape', type=int, nargs="+", default=[60, 90])
parser.add_argument('--heatmap-scale', type=float, default=1)
parser.add_argument('--data-format', type=str, default='NCHW')


def main(args):
    # Get dataset
    datasource = DataSource(None, [args.input_image], shape=tuple(args.eye_shape),
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
    model = CNN(datasource.tensors, datasource.x_shape, learning_schedule,
                data_format=args.data_format)
    
    # Get evaluator
    evaluator = Trainer(model, model_checkpoint=args.model_checkpoint)

    # Predict
    output, losses = evaluator.run_predict(datasource)
    input_data = util.load_pickle(args.input_image)
    print('Losses', losses)
    util.plot_predictions(output, input_data, tuple(args.eye_shape))


if __name__ == '__main__':
    main(parser.parse_args())