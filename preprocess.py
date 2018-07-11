#!/usr/bin/env python3
"""Main script for training a model for gaze estimation."""
import argparse

import coloredlogs
import tensorflow as tf

from preprocessing.unityeyes import UnityEyes
from models.cnn import CNN

if __name__ == '__main__':
    unityeyes = UnityEyes(
        data_format='NCHW',
        images_path='datasets',
        generate_heatmaps=True,
        eye_image_shape=(36, 60),
        heatmaps_scale=1.0
    )

    unityeyes.set_augmentation_range('translation', 2.0, 10.0)
    unityeyes.set_augmentation_range('rotation', 1.0, 10.0)
    unityeyes.set_augmentation_range('intensity', 0.5, 20.0)
    unityeyes.set_augmentation_range('blur', 0.1, 1.0)
    unityeyes.set_augmentation_range('scale', 0.01, 0.1)
    unityeyes.set_augmentation_range('rescale', 1.0, 0.5)
    unityeyes.set_augmentation_range('num_line', 0.0, 2.0)
    unityeyes.set_augmentation_range('heatmap_sigma', 7.5, 2.5)

    unityeyes.preprocess_data()


