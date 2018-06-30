#!/usr/bin/env python3
"""Main script for training a model for gaze estimation."""
import argparse

import coloredlogs
import tensorflow as tf

from preprocessing.unityeyes import UnityEyes
from models.cnn import ELG

if __name__ == '__main__':
    
    unityeyes = UnityEyes(
        None,
        data_format='NCHW',
        images_path='datasets',
        batch_size=32,
        # min_after_dequeue=1000,
        generate_heatmaps=True,
        #shuffle=True,
        #staging=True,
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


