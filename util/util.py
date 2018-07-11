import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
import tensorflow as tf


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