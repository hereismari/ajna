from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

import util.util as util

from data_sources.data_source import DataSource
from models.cnn import CNN

from learning.trainer import Trainer

import glob

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Train CNN (Elg).')
parser.add_argument('--input-image', type=str, required=True)
parser.add_argument('--model-checkpoint', type=str, required=True)


def plot_predictions(output, input_data):
    eye = input_data['eye'].reshape(36, 60)

    fig, ax = plt.subplots(1, 3)

    # in same graph
    img = ax[0].imshow(eye, cmap='gray')

    landmarks = input_data['landmarks'].reshape(18, 2)
    for landmark in landmarks:
        circ = Circle((landmark[0], landmark[1]), 1, alpha=0.7, color='green')
        ax[0].add_patch(circ)
    
    landmarks = output['landmarks'].reshape(18, 2)
    for landmark in landmarks:
        circ = Circle((landmark[0], landmark[1]), 1, alpha=0.7, color='red')
        ax[0].add_patch(circ)
    ax[0].set_title('Predições Vs Pontos reais')
    # real
    img = ax[1].imshow(eye, cmap='gray')

    landmarks = input_data['landmarks'].reshape(18, 2)
    for landmark in landmarks:
        circ = Circle((landmark[0], landmark[1]), 1, alpha=0.7, color='green')
        ax[1].add_patch(circ)

    ax[1].set_title('Pontos reais')
    # predicted
    img = ax[2].imshow(eye, cmap='gray')

    landmarks = output['landmarks'].reshape(18, 2)
    for landmark in landmarks:
        circ = Circle((landmark[0], landmark[1]), 1, alpha=0.7, color='red')
        ax[2].add_patch(circ)

    ax[2].set_title('Predições')
    plt.show()


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
        }
    ]
    model = CNN(datasource.tensors, datasource.x_shape, learning_schedule)
    
    # Get evaluator
    evaluator = Trainer(model)

    # Predict
    output, losses = evaluator.run_predict(datasource)
    input_data = util.load_pickle(args.input_image)
    print('Losses', losses)
    plot_predictions(output, input_data)


if __name__ == '__main__':
    main(parser.parse_args())