'''
Based on: https://github.com/swook/GazeML
Modified by: @mari-linhares
'''

import tensorflow as tf
import numpy as np

class CNN(object):
    def __init__(self, data_holder, input_shape, learning_schedule, data_format='NCHW', predict_only=False):
        self._training = tf.Variable(True, dtype=tf.bool, trainable=False)
        self._train = tf.assign(self._training, True)
        self._eval = tf.assign(self._training, False)

        self.train_count = tf.Variable(0.0, trainable=False)
        self.increase_train_count = tf.assign_add(self.train_count, 1)

        self.define_data(data_holder, predict_only)

        self._hg_first_layer_stride = 1
        self._hg_num_modules = 3
        self._hg_num_feature_maps = 32
        self._hg_num_landmarks = 18
        self._hg_num_residual_blocks = 1

        # NCHW
        self._data_format_longer = 'channels_first' if data_format == 'NCHW' else 'channels_last'
        print(data_format)
        self._data_format = data_format
        self.use_batch_statistics = True
        self.is_training = True

        self._learning_schedule = learning_schedule

        self.get_model(predict_only)


    @staticmethod
    def _tf_mse(x, y):
        """Tensorflow call for mean-squared error."""
        return tf.reduce_mean(tf.squared_difference(x, y))


    def train_iteration(self, sess, *args, **kwargs):
        sess.run(self.increase_train_count)
        return sess.run(self.backprop)
    
    def run_model(self, sess):
        return sess.run([self.X, self.landmarks])

    def eval_iteration(self, sess):
        return sess.run(self.run_eval)

    def get_model(self, predict_only=False):
        X_pred, losses, metrics = self.build_model(predict_only)
        
        self.model = X_pred
        self.losses = losses
        self.metrics = metrics

        # Summaries
        if not predict_only:
            heatmap_summary = tf.summary.scalar("heatmaps_mse", self.losses['heatmaps_mse'])
            radius_summary = tf.summary.scalar("radius_mse", self.losses['radius_mse'])
            self.summaries = tf.summary.merge_all()

            self.train_writer = tf.summary.FileWriter('checkpoints/train')
            self.eval_writer = tf.summary.FileWriter('checkpoints/test')

            self.run_eval = [self.summaries, self.model, self.losses]
        else:
            self.run_eval = [self.model, self.model, self.model]
        if not predict_only:
            self.build_optimizer()

    def define_data(self, data_holder, predict_only=False):
        if predict_only:
            self.X = data_holder
        else:
            self.X = data_holder[0]

            self.Y1 = data_holder[1]
            self.Y2 = data_holder[2]
            self.Y3 = data_holder[3]
    
    def build_optimizer(self):
        self._build_optimizers()
        self.backprop = [self.summaries] + self._optimize_ops + [self.losses]
    
    def _build_optimizers(self):
        """Based on learning schedule, create optimizer instances."""
        self._optimize_ops = []
        all_trainable_variables = tf.trainable_variables()
        for spec in self._learning_schedule:
            optimize_ops = []
            loss_terms = spec['loss_terms_to_optimize']
            assert isinstance(loss_terms, dict)
            for loss_term_key, prefixes in loss_terms.items():
                variables_to_train = []
                for prefix in prefixes:
                    variables_to_train += [
                        v for v in all_trainable_variables
                        if v.name.startswith(prefix)
                    ]
                optimize_op = tf.train.AdamOptimizer(
                    learning_rate=spec['learning_rate'],
                ).minimize(
                    loss=self.losses[loss_term_key],
                    var_list=variables_to_train,
                    name='optimize_%s' % loss_term_key,
                )
                optimize_ops.append(optimize_op)
            self._optimize_ops.append(optimize_ops)
            print('Built optimizer for: %s' % ', '.join(loss_terms.keys()))

    def build_model(self, predict_only=False):
        outputs = {}
        loss_terms = {}
        metrics = {}

        with tf.variable_scope('hourglass'):
            # Prepare for Hourglass by downscaling via conv
            with tf.variable_scope('pre'):
                n = self._hg_num_feature_maps
                x = self._apply_conv(self.X, num_features=n, kernel_size=7,
                                     stride=self._hg_first_layer_stride)
                x = tf.nn.relu(self._apply_bn(x))
                x = self._build_residual_block(x, n, 2*n, name='res1')
                x = self._build_residual_block(x, 2*n, n, name='res2')

            # Hourglass blocks
            x_prev = x
            for i in range(self._hg_num_modules):
                with tf.variable_scope('hg_%d' % (i + 1)):
                    x = self._build_hourglass(x, steps_to_go=4, num_features=self._hg_num_feature_maps)
                    x, h = self._build_hourglass_after(
                        x_prev, x, do_merge=(i < (self._hg_num_modules - 1)),
                    )
                    
                    if not predict_only:
                        metrics['heatmap%d_mse' % (i + 1)] = CNN._tf_mse(h, self.Y1)
                    else:
                        metrics['heatmap%d_mse' % (i + 1)] = None
                    x_prev = x
            
            if not predict_only:
                loss_terms['heatmaps_mse'] = tf.reduce_mean([
                    metrics['heatmap%d_mse' % (i + 1)] for i in range(self._hg_num_modules)
                ])
            else:
                loss_terms['heatmaps_mse'] = None
            x = h
            outputs['heatmaps'] = x

        # Soft-argmax
        x = self._calculate_landmarks(x)
        self.landmarks = x

        with tf.variable_scope('upscale'):
            # Upscale since heatmaps are half-scale of original image
            x *= self._hg_first_layer_stride

            if not predict_only:
                metrics['landmarks_mse'] = CNN._tf_mse(x, self.Y2)
            else:
                metrics['landmarks_mse'] = None
            
            outputs['landmarks'] = x

        # Fully-connected layers for radius regression
        with tf.variable_scope('radius'):
            x = tf.contrib.layers.flatten(tf.transpose(x, perm=[0, 2, 1]))
            for i in range(3):
                with tf.variable_scope('fc%d' % (i + 1)):
                    x = tf.nn.relu(self._apply_bn(self._apply_fc(x, 100)))
            with tf.variable_scope('out'):
                x = self._apply_fc(x, 1)
            outputs['radius'] = x
            
            if not predict_only:
                metrics['radius_mse'] = CNN._tf_mse(tf.reshape(x, [-1]), self.Y3)
                loss_terms['radius_mse'] = 1e-7 * metrics['radius_mse']
            else:
                metrics['radius_mse'] = None
                loss_terms['radius_mse'] = None
        
        # Define outputs
        return outputs, loss_terms, metrics

    def _apply_conv(self, tensor, num_features, kernel_size=3, stride=1):
        return tf.layers.conv2d(
            tensor,
            num_features,
            kernel_size=kernel_size,
            strides=stride,
            padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_initializer=tf.zeros_initializer(),
            data_format=self._data_format_longer,
            name='conv',
        )

    def _apply_fc(self, tensor, num_outputs):
        return tf.layers.dense(
            tensor,
            num_outputs,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_initializer=tf.zeros_initializer(),
            name='fc',
        )

    def _apply_pool(self, tensor, kernel_size=3, stride=2):
        tensor = tf.layers.max_pooling2d(
            tensor,
            pool_size=kernel_size,
            strides=stride,
            padding='SAME',
            data_format=self._data_format_longer,
            name='pool',
        )
        return tensor

    def _apply_bn(self, tensor):
        return tf.contrib.layers.batch_norm(
            tensor,
            scale=True,
            center=True,
            is_training=self.use_batch_statistics,
            trainable=True,
            data_format=self._data_format,
            updates_collections=None,
        )

    def _build_residual_block(self, x, num_in, num_out, name='res_block'):
        with tf.variable_scope(name):
            half_num_out = max(int(num_out/2), 1)
            c = x
            with tf.variable_scope('conv1'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=half_num_out, kernel_size=1, stride=1)
            with tf.variable_scope('conv2'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=half_num_out, kernel_size=3, stride=1)
            with tf.variable_scope('conv3'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=num_out, kernel_size=1, stride=1)
            with tf.variable_scope('skip'):
                if num_in == num_out:
                    s = tf.identity(x)
                else:
                    s = self._apply_conv(x, num_features=num_out, kernel_size=1, stride=1)
            x = c + s
        return x

    def _build_hourglass(self, x, steps_to_go, num_features, depth=1):
        with tf.variable_scope('depth%d' % depth):
            # Upper branch
            up1 = x
            for i in range(self._hg_num_residual_blocks):
                up1 = self._build_residual_block(up1, num_features, num_features,
                                                 name='up1_%d' % (i + 1))
            # Lower branch
            low1 = self._apply_pool(x, kernel_size=2, stride=2)
            for i in range(self._hg_num_residual_blocks):
                low1 = self._build_residual_block(low1, num_features, num_features,
                                                  name='low1_%d' % (i + 1))
            # Recursive
            low2 = None
            if steps_to_go > 1:
                low2 = self._build_hourglass(low1, steps_to_go - 1, num_features, depth=depth+1)
            else:
                low2 = low1
                for i in range(self._hg_num_residual_blocks):
                    low2 = self._build_residual_block(low2, num_features, num_features,
                                                      name='low2_%d' % (i + 1))
            # Additional residual blocks
            low3 = low2
            for i in range(self._hg_num_residual_blocks):
                low3 = self._build_residual_block(low3, num_features, num_features,
                                                  name='low3_%d' % (i + 1))
            # Upsample
            if self._data_format == 'NCHW':  # convert to NHWC
                low3 = tf.transpose(low3, (0, 2, 3, 1))
            up2 = tf.image.resize_bilinear(
                    low3,
                    up1.shape[1:3] if self._data_format == 'NHWC' else up1.shape[2:4],
                    align_corners=True,
                  )
            if self._data_format == 'NCHW':  # convert back from NHWC
                up2 = tf.transpose(up2, (0, 3, 1, 2))

        return up1 + up2

    def _build_hourglass_after(self, x_prev, x_now, do_merge=True):
        with tf.variable_scope('after'):
            for j in range(self._hg_num_residual_blocks):
                x_now = self._build_residual_block(x_now, self._hg_num_feature_maps,
                                                   self._hg_num_feature_maps,
                                                   name='after_hg_%d' % (j + 1))
            x_now = self._apply_conv(x_now, self._hg_num_feature_maps, kernel_size=1, stride=1)
            x_now = self._apply_bn(x_now)
            x_now = tf.nn.relu(x_now)

            with tf.variable_scope('hmap'):
                h = self._apply_conv(x_now, self._hg_num_landmarks, kernel_size=1, stride=1)

        x_next = x_now
        if do_merge:
            with tf.variable_scope('merge'):
                with tf.variable_scope('h'):
                    x_hmaps = self._apply_conv(h, self._hg_num_feature_maps, kernel_size=1, stride=1)
                with tf.variable_scope('x'):
                    x_now = self._apply_conv(x_now, self._hg_num_feature_maps, kernel_size=1, stride=1)
                x_next += x_prev + x_hmaps
        return x_next, h

    _softargmax_coords = None

    def _calculate_landmarks(self, x):
        """Estimate landmark location from heatmaps."""
        with tf.variable_scope('argsoftmax'):
            if self._data_format == 'NHWC':
                _, h, w, _ = x.shape.as_list()
            else:
                _, _, h, w = x.shape.as_list()
            if self._softargmax_coords is None:
                # Assume normalized coordinate [0, 1] for numeric stability
                ref_xs, ref_ys = np.meshgrid(np.linspace(0, 1.0, num=w, endpoint=True),
                                             np.linspace(0, 1.0, num=h, endpoint=True),
                                             indexing='xy')
                ref_xs = np.reshape(ref_xs, [-1, h*w])
                ref_ys = np.reshape(ref_ys, [-1, h*w])
                self._softargmax_coords = (
                    tf.constant(ref_xs, dtype=tf.float32),
                    tf.constant(ref_ys, dtype=tf.float32),
                )
            ref_xs, ref_ys = self._softargmax_coords

            # Assuming N x 18 x 45 x 75 (NCHW)
            beta = 1e2
            if self._data_format == 'NHWC':
                x = tf.transpose(x, (0, 3, 1, 2))
            x = tf.reshape(x, [-1, self._hg_num_landmarks, h*w])
            x = tf.nn.softmax(beta * x, axis=-1)
            lmrk_xs = tf.reduce_sum(ref_xs * x, axis=[2])
            lmrk_ys = tf.reduce_sum(ref_ys * x, axis=[2])

            # Return to actual coordinates ranges
            return tf.stack([
                lmrk_xs * (w - 1.0) + 0.5,
                lmrk_ys * (h - 1.0) + 0.5,
            ], axis=2)  # N x 18 x 2

    def train(self, sess, train=True):
        if train:
            sess.run(self._train)
        else:
            sess.run(self._eval)

    def eval(self, sess):
        self.train(sess, train=False)