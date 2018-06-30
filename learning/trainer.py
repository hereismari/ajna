import tensorflow as tf
import numpy as np
import csv

from util import util

class Trainer(object):
    def __init__(self, model):
        self.model = model

        self.eval_steps = 100
        self.exec_name = 'train'
        self.running_losses = {}

    def run_training(self, data, max_steps, eval=True, test=True):
        self.max_steps = max_steps
        with tf.Session() as sess:
            self.initialize_vars(sess)
            self.train(sess, data, eval=eval)

    def initialize_vars(self, sess):
        self.running_loss, self.running_steps = 0, 0

        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        sess.run(init)
        sess.run(init_l)

    def train(self, sess, data, eval=True):
        self.data = data
        data.run(sess)
        for self.step in range(self.max_steps):
            self.train_step(sess, data, eval=eval)

    def train_step(self, sess, data, eval=True):
        self.train_batch(sess, data)
        self._eval_step(sess, data)

    def train_batch(self, sess, data):
        self.model.train(sess)
        _, losses = self.model.train_iteration(sess, data)
        for key in losses:
            if key not in self.running_losses:
                self.running_losses[key] = 0
            self.running_losses[key] += losses[key]
        self.running_steps += 1

    def _eval_step(self, sess, data):
        if eval and (self.step % self.eval_steps == 0) or (self.step + 1 == self.max_steps):
            self.model.eval(sess)
            self.eval_step(sess, data)
            data.run(sess)

    def eval_step(self, sess, data):
        s = ''
        for loss in self.running_losses:
            s += 'Running Loss %s: %g' % (loss, self.running_losses[loss] / self.running_steps)
            s += '  |  '
        self.print_progress(s)
        self.running_losses = {}
        self.running_steps = 0

    def print_progress(self, status):
        util.print_progress_bar(self.step + 1, self.max_steps, status)