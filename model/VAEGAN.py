
import tensorflow as tf


class VAEGAN(object):
    def __init__(self, z_len):
        """
        :type output_dist: Distribution
        :type latent_spec: list[(Distribution, bool)]
        :type batch_size: int
        :type network_type: string
        """
        self.initialize = tf.random_normal_initializer(mean=0, stddev=0.02)
        self.mode = tf.placeholder(dtype=tf.bool, name="mode")
        self.z_len = z_len

    def encoder(self, image, reuse=None):

        with tf.variable_scope('encoder', reuse=reuse):
            conv_1 = self._conv_layer(image, 5, 2, 64, 'conv_1')
            conv_2 = self._conv_layer(conv_1, 5, 2, 128, 'conv_2')
            conv_3 = self._conv_layer(conv_2, 5, 2, 256, 'conv_3')

            flatten = tf.layers.flatten(conv_3, 'flat')

            enc_dense = tf.layers.dense(flatten, 1024, kernel_initializer=self.initialize, name='enc_dense')
            enc_dense_bn = self._batch_normalization(enc_dense, name="enc_dense_bn")
            enc_dense_act = tf.nn.relu(enc_dense_bn, name='enc_dense_act')

            mu = tf.layers.dense(enc_dense_act, self.z_len, kernel_initializer=self.initialize, name='mu')
            logvar = tf.layers.dense(enc_dense_act, self.z_len, kernel_initializer=self.initialize, name='logvar')
            stddev = tf.exp(0.5*logvar, name='stddev')
            epsilon = tf.random_normal(tf.shape(mu))
            enc_out = epsilon * stddev + mu

            return enc_out, mu, logvar

    def generator(self, z_var, reuse=None):

        with tf.variable_scope('generator', reuse=reuse):
            gen_dense = tf.layers.dense(z_var, 8 * 8 * 256, kernel_initializer=self.initialize, name='gen_dense')
            gen_dense_bn = self._batch_normalization(gen_dense, name='gen_dense_bn')
            gen_dense_act = tf.nn.relu(gen_dense_bn, name='gen_dense_act')
            gen_dense_reshape = tf.reshape(gen_dense_act, [-1, 8, 8, 256], name='gen_dense_reshape')

            deconv_1 = self._deconv_layer(gen_dense_reshape, 5, 2, 256, 'deconv_1')
            deconv_2 = self._deconv_layer(deconv_1, 5, 2, 128, 'deconv_2')
            deconv_3 = self._deconv_layer(deconv_2, 5, 2, 32, 'deconv_3')
            gen_output = self._deconv_layer(deconv_3, 5, 1, 3, 'gen_output', 'tanh', bn=False)

            return gen_output

    def discriminator(self, dis_images, reuse=None):

        with tf.variable_scope('discriminator', reuse=reuse):

            dis_conv1 = self._conv_layer(dis_images, 5, 1, 32, 'dis_conv1', bn=False)
            dis_conv2 = self._conv_layer(dis_conv1, 5, 2, 128, 'dis_conv2')
            dis_conv3 = self._conv_layer(dis_conv2, 5, 2, 256, 'dis_conv3')
            dis_conv4 = self._conv_layer(dis_conv3, 5, 2, 256, 'dis_conv4', act='None', bn=False)
            dis_conv4_bn_act = tf.nn.relu(self._batch_normalization(dis_conv4,
                                                                    name='dis_conv4_bn'), name='dis_conv4_act')

            flatten = tf.layers.flatten(dis_conv4_bn_act, 'dis_flat')
            #feat_flatten = tf.layers.flatten(dis_conv4, 'feat_flat')

            dis_dense = tf.layers.dense(flatten, 512, kernel_initializer=self.initialize, name='dis_dense')
            dis_dense_bn = self._batch_normalization(dis_dense, name='dis_dense_bn')
            dis_dense_act = tf.nn.relu(dis_dense_bn, name='dis_dense_act')

            dis_out = tf.layers.dense(dis_dense_act, 1, kernel_initializer=self.initialize, name='dis_out')

            return tf.nn.sigmoid(dis_out), dis_conv4

    def _conv_layer(self, input, filter_size, stride, out_channels, name, act='relu', bn=True):
        shape = input.get_shape()
        with tf.variable_scope(name):
            weights = tf.get_variable(name="filter", initializer=self.initialize,
                                      shape=[filter_size, filter_size, shape[-1], out_channels],
                                      dtype=tf.float32)
            bias = tf.get_variable(name="bias", initializer=tf.zeros_initializer, shape=[out_channels],
                                   dtype=tf.float32)
            conv = tf.nn.conv2d(input, weights, [1, stride, stride, 1], padding='SAME')

            bias_add = tf.nn.bias_add(conv, bias)

            if bn:
                batch_norm = self._batch_normalization(bias_add, name + '_bn')
            else:
                batch_norm = bias_add

            print("activation of layer {} : {}".format(name, act))

            if act == 'relu':
                activation = tf.nn.relu(batch_norm)
            elif act == 'lrelu':
                activation = tf.nn.leaky_relu(batch_norm)
            elif act == 'sigmoid':
                activation = tf.nn.sigmoid(batch_norm)
            else:
                activation = batch_norm
            return activation

    def _deconv_layer(self, input, filter_size, stride, out_channels, name, act='relu', bn=True):
        shape = input.get_shape()
        height = stride * shape[1].value
        width = stride * shape[2].value
        with tf.variable_scope(name):
            weights = tf.get_variable(name="filter", initializer=self.initialize,
                                      shape=[filter_size, filter_size, out_channels, shape[-1]],
                                      dtype=tf.float32)
            bias = tf.get_variable(name="bias", initializer=tf.zeros_initializer, shape=[out_channels],
                                   dtype=tf.float32)
            deconv = tf.nn.conv2d_transpose(input, weights, [tf.shape(input)[0], height, width, out_channels],
                                            [1, stride, stride, 1], padding='SAME')

            bias_add = tf.nn.bias_add(deconv, bias)

            if bn:
                batch_norm = self._batch_normalization(bias_add, name + '_bn')
            else:
                batch_norm = bias_add

            print("activation of layer {} : {}".format(name, act))

            if act == 'relu':
                activation = tf.nn.relu(batch_norm)
            elif act == 'lrelu':
                activation = tf.nn.leaky_relu(batch_norm)
            elif act == 'tanh':
                activation = tf.nn.tanh(batch_norm)
            else:
                activation = batch_norm
            return activation

    def _batch_normalization(self, x, name):

        return tf.contrib.layers.batch_norm(x, decay=0.95, center=True, scale=True, is_training=self.mode,
                                            epsilon=1e-8, updates_collections=None, scope=name)
