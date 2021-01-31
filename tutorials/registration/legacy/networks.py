# (Provided) This uses TensorFlow
# *** Available as part of UCL MPHY0025 (Information Processing in Medical Imaging) Assessed Coursework 2018-19 ***
# *** This code is with an Apache 2.0 license, University College London ***

import tensorflow as tf


def var_conv_kernel(ch_in, ch_out, name='W', initialiser=None):
    with tf.variable_scope(name):
        k_conv = [3, 3]
        if initialiser is None:
            initialiser = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name, shape=k_conv+[ch_in]+[ch_out], initializer=initialiser)


def conv_block(input_, ch_in, ch_out, strides=None, name='conv_block'):
    if strides is None:
        strides = [1, 1, 1, 1]
    with tf.variable_scope(name):
        w = var_conv_kernel(ch_in, ch_out)
        return tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(input_, w, strides, "SAME")))


def deconv_block(input_, ch_in, ch_out, shape_out, strides, name='deconv_block'):
    with tf.variable_scope(name):
        w = var_conv_kernel(ch_out, ch_in)
        return tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d_transpose(input_, w, shape_out, strides, "SAME")))


def downsample_resnet_block(input_, ch_in, ch_out, name='down_resnet_block'):
    strides1 = [1, 1, 1, 1]
    strides2 = [1, 2, 2, 1]
    k_pool = [1, 2, 2, 1]
    with tf.variable_scope(name):
        h0 = conv_block(input_, ch_in, ch_out, name='W0')
        r1 = conv_block(h0, ch_out, ch_out, name='WR1')
        wr2 = var_conv_kernel(ch_out, ch_out)
        r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(r1, wr2, strides1, "SAME")) + h0)
        h1 = tf.nn.max_pool(r2, k_pool, strides2, padding="SAME")
        return h1, h0


def upsample_resnet_block(input_, input_skip, ch_in, ch_out, name='up_resnet_block'):
    strides1 = [1, 1, 1, 1]
    strides2 = [1, 2, 2, 1]
    size_out = input_skip.shape.as_list()
    with tf.variable_scope(name):
        h0 = deconv_block(input_, ch_in, ch_out, size_out, strides2)
        r1 = h0 + input_skip
        r2 = conv_block(h0, ch_out, ch_out)
        wr2 = var_conv_kernel(ch_out, ch_out)
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(r2, wr2, strides1, "SAME")) + r1)
        return h1


def ddf_output(input_, ch_in, name='ddf_summand'):
    strides1 = [1, 1, 1, 1]
    initial_std_local = 0.01
    initial_bias_local = 0.01
    with tf.variable_scope(name):
        w = var_conv_kernel(ch_in, 2, initialiser=tf.random_normal_initializer(0, initial_std_local))
        b = tf.get_variable(name, shape=[2], initializer=tf.constant_initializer(initial_bias_local))
        return tf.nn.conv2d(input_, w, strides1, "SAME") + b


def get_reference_grid(size):
    return tf.to_float(tf.stack(tf.meshgrid([i for i in range(size[1])], [j for j in range(size[0])]), axis=2))


class RegNet2D:

    def __init__(self, minibatch_size, image_moving, image_fixed):

        self.num_channel_initial = 32

        self.minibatch_size = minibatch_size
        self.grid_ref = get_reference_grid(image_fixed.shape.as_list()[1:3])
        self.image_moving = image_moving
        self.input_layer = tf.concat([self.image_moving, image_fixed], axis=3)

        nc = [int(self.num_channel_initial*(2**i)) for i in range(5)]
        h0, hc0 = downsample_resnet_block(self.input_layer, 2, nc[0], name='local_down_0')
        h1, hc1 = downsample_resnet_block(h0, nc[0], nc[1], name='local_down_1')
        h2, hc2 = downsample_resnet_block(h1, nc[1], nc[2], name='local_down_2')
        h3, hc3 = downsample_resnet_block(h2, nc[2], nc[3], name='local_down_3')
        h4 = conv_block(h3, nc[3], nc[4], name='local_deep_4')
        hu3 = upsample_resnet_block(h4, hc3, nc[4], nc[3], name='local_up_3')
        hu2 = upsample_resnet_block(hu3, hc2, nc[3], nc[2], name='local_up_2')
        hu1 = upsample_resnet_block(hu2, hc1, nc[2], nc[1], name='local_up_1')
        hu0 = upsample_resnet_block(hu1, hc0, nc[1], nc[0], name='local_up_0')

        self.ddf = ddf_output(hu0, nc[0])
        self.grid_warped = self.grid_ref + self.ddf

    def warp_image(self, input_=None):
        if input_ is None:
            input_ = self.image_moving
        return tf.contrib.resampler.resampler(input_, self.grid_warped)
