import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

import cv2
import numpy as np

from collections import OrderedDict


def print_progress_bar(count, total, status='', bar_len=50, verbose=True, log_file=None):
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    if verbose:
        print('(%5d/%5d) [%s] %s%s \t %s' % (count, total, bar, percents, '%', status))
    else:
        print_replace('(%5d/%5d) [%s] %s%s \t %s' % (count, total, bar, percents, '%', status))
    
    if log_file:
        log_file.write('(%5d/%5d) [%s] %s%s \t %s' % (count, total, bar, percents, '%', status))


def print_replace(status):
    ERASE_LINE = '\x1b[2K'
    print('\r%s' % (ERASE_LINE), end = '', flush=True)
    print('\r%s' % (status), end = '', flush=True)


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_pickle(filename):
     return pickle.load(open(filename, 'rb'))


def print_progress_bar(count, total, status='', bar_len=50, verbose=True, log_file=None):
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    if verbose:
        print('(%5d/%5d) [%s] %s%s \t %s' % (count, total, bar, percents, '%', status))
    else:
        print_replace('(%5d/%5d) [%s] %s%s \t %s' % (count, total, bar, percents, '%', status))
    
    if log_file:
        log_file.write('(%5d/%5d) [%s] %s%s \t %s' % (count, total, bar, percents, '%', status))


def print_replace(status):
    ERASE_LINE = '\x1b[2K'
    print('\r%s' % (ERASE_LINE), end = '', flush=True)
    print('\r%s' % (status), end = '', flush=True)


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

        
def load_pickle(filename):
     return pickle.load(open(filename, 'rb'))
    
    
def get_basename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

  
def plot_predictions(output, input_data, eye_shape):
    eye = input_data['eye'].reshape(*eye_shape)

    fig, ax = plt.subplots(1, 3)

    # both in same image
    img = ax[0].imshow(eye, cmap='gray')
    ax[0] = _plot_predictions(ax[0], input_data, colors=['blue'] * 3)
    ax[0] = _plot_predictions(ax[0], output, colors=['red'] * 3)
    ax[0].set_title('Predições Vs Pontos reais')
    
    # real
    img = ax[1].imshow(eye, cmap='gray')
    ax[1] = _plot_predictions(ax[1], input_data)
    ax[1].set_title('Pontos reais')
    
    # predicted
    img = ax[2].imshow(eye, cmap='gray')
    ax[2] = _plot_predictions(ax[2], output)
    ax[2].set_title('Predições')
    plt.show()


def _plot_predictions(ax, data, colors=['purple', 'orange',  'magenta']):
    eye_landmarks = data['landmarks'].reshape(18, 2)

    landmarks = OrderedDict()
    landmarks['eyelid'] = eye_landmarks[0:8, :]
    landmarks['iris'] = eye_landmarks[8:16, :]
    landmarks['iris_centre'] = eye_landmarks[16, :].reshape(1, 2)
    
    for key, color in zip(landmarks, colors):
        landmark = landmarks[key]
        for points in landmark:
            circ = Circle((points[0], points[1]), 1, alpha=0.7, color=color)
            ax.add_patch(circ)

    return ax


def plot_predictions2(output, input_img):
    eye = input_img
    output_landmarks = output.reshape(18, 2)

    fig, ax = plt.subplots(1, 1)

    # both in same image
    img = ax.imshow(eye, cmap='gray')
    for landmark in output_landmarks:
        circ = Circle((landmark[0], landmark[1]), 1, alpha=0.7, color='red')
        ax.add_patch(circ)

    plt.show(), plt.pause(2)



def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos.astype(np.float32)).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out