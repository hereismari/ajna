import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


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
    

def plot_predictions(output, input_data, eye_shape):
    eye = input_data['eye'].reshape(*eye_shape)
    input_landmarks = input_data['landmarks'].reshape(18, 2)
    output_landmarks = output['landmarks'].reshape(18, 2)

    fig, ax = plt.subplots(1, 3)

    # both in same image
    img = ax[0].imshow(eye, cmap='gray')

    for landmark in input_landmarks:
        circ = Circle((landmark[0], landmark[1]), 1, alpha=0.7, color='green')
        ax[0].add_patch(circ)
    for landmark in output_landmarks:
        circ = Circle((landmark[0], landmark[1]), 1, alpha=0.7, color='red')
        ax[0].add_patch(circ)
    ax[0].set_title('Predições Vs Pontos reais')
    
    # real
    img = ax[1].imshow(eye, cmap='gray')
    for landmark in input_landmarks:
        circ = Circle((landmark[0], landmark[1]), 1, alpha=0.7, color='green')
        ax[1].add_patch(circ)
    ax[1].set_title('Pontos reais')
    
    # predicted
    img = ax[2].imshow(eye, cmap='gray')
    for landmark in output_landmarks:
        circ = Circle((landmark[0], landmark[1]), 1, alpha=0.7, color='red')
        ax[2].add_patch(circ)

    ax[2].set_title('Predições')
    plt.show()
