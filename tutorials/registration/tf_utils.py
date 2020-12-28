# *** This code is with an Apache 2.0 license, University College London ***

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

### network and layers
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
        return bilinear_resampler(input_, self.grid_warped)


### transformation
def get_reference_grid(grid_size):
    # grid_size: [batch_size, height, width]
    grid = tf.cast(tf.stack(tf.meshgrid(
                        tf.range(grid_size[1]),
                        tf.range(grid_size[2]),
                        indexing='ij'), axis=2), dtype=tf.float32)
    return tf.tile(tf.expand_dims(grid, axis=0), [grid_size[0],1,1,1])


def warp_grid(grid, transform):
    # grid: [batch, height, width, 2]
    # transform: [batch, 3, 3]
    batch_size, height, width = grid.shape[0:3]
    grid = tf.concat([tf.reshape(grid,[batch_size,height*width,2]), 
                    tf.ones([batch_size,height*width,1])], axis=2)
    grid_warped = tf.matmul(grid, transform)
    return tf.reshape(grid_warped[...,:2], [batch_size,height,width,2])


def bilinear_resampler(grid_data, sample_grids):
    '''
    grid_data: [batch, height, width]
    sample_grids: [batch, height, width, 2]    
    '''
    batch_size, height, width = (grid_data.shape[:])
    sample_coords = tf.reshape(sample_grids, [batch_size,-1,2])
    # pad to replicate the boundaries 1-ceiling, 2-floor
    sample_coords = tf.stack([tf.clip_by_value(sample_coords[...,0],0,height-1),
                            tf.clip_by_value(sample_coords[...,1],0,width-1)], axis=2)
    i1 = tf.cast(tf.math.ceil(sample_coords[...,0]), dtype=tf.int32)
    j1 = tf.cast(tf.math.ceil(sample_coords[...,1]), dtype=tf.int32)
    i0 = tf.maximum(i1-1, 0)
    j0 = tf.maximum(j1-1, 0)
    # four data points q_ij
    q00 = tf.gather_nd(grid_data, tf.stack([i0,j0],axis=2), batch_dims=1)
    q01 = tf.gather_nd(grid_data, tf.stack([i0,j1],axis=2), batch_dims=1)
    q11 = tf.gather_nd(grid_data, tf.stack([i1,j1],axis=2), batch_dims=1)
    q10 = tf.gather_nd(grid_data, tf.stack([i1,j0],axis=2), batch_dims=1)    
    # weights with normalised local coordinates
    wi1 = sample_coords[...,0] - tf.cast(i0,dtype=tf.float32)
    wi0 = 1 - wi1
    wj1 = sample_coords[...,1] - tf.cast(j0,dtype=tf.float32)
    wj0 = 1 - wj1
    return tf.reshape(q00*wi0*wj0 + q01*wi0*wj1 + q11*wi1*wj1 + q10*wi1*wj0, [batch_size,height,width])


def random_transform_generator(batch_size, corner_scale=.1):
    # right-multiplication affine
    ori_corners = tf.tile([[[1.,1.], [1.,-1.], [-1.,1.], [-1.,-1.]]], [batch_size,1,1])
    new_corners = ori_corners + tf.random.uniform([batch_size,4,2], -corner_scale, corner_scale)    
    ori_corners = tf.concat([ori_corners,tf.ones([batch_size,4,1])], axis=2)
    new_corners = tf.concat([new_corners,tf.ones([batch_size,4,1])], axis=2)
    return tf.stack([tf.linalg.lstsq(ori_corners[n],new_corners[n]) for n in range(batch_size)], axis=0)


def random_image_transform(images):
    # images: [batch_size, height, width]
    reference_grid = get_reference_grid(images.shape[0:3])
    random_transform = random_transform_generator(images.shape[0], corner_scale=0.1)
    sample_grids = warp_grid(reference_grid, random_transform)
    return bilinear_resampler(images, sample_grids)


### loss functions
def normalised_cross_correlation(ts, ps, eps=0.0):
    dp = ps - tf.reduce_mean(ps, axis=[1, 2, 3])
    dt = ts - tf.reduce_mean(ts, axis=[1, 2, 3])
    vp = tf.reduce_sum(tf.square(dp), axis=[1, 2, 3])
    vt = tf.reduce_sum(tf.square(dt), axis=[1, 2, 3])
    return tf.constant(1.0) - tf.reduce_sum(dp*dt / (tf.sqrt(vp*vt) + eps), axis=[1, 2, 3])


def normalised_cross_correlation2(ts, ps, eps=1e-6):
    mean_t = tf.reduce_mean(ts, axis=[1, 2, 3])
    mean_p = tf.reduce_mean(ps, axis=[1, 2, 3])
    std_t = tf.reduce_sum(tf.sqrt(tf.square(mean_t)-tf.reduce_mean(tf.square(ts), axis=[1, 2, 3])), axis=[1, 2, 3])
    std_p = tf.reduce_sum(tf.sqrt(tf.square(mean_p)-tf.reduce_mean(tf.square(ps), axis=[1, 2, 3])), axis=[1, 2, 3])
    return -tf.reduce_mean((ts-mean_t)*(ps-mean_p) / (std_t*std_p+eps), axis=[1, 2, 3])


def sum_square_difference(i1, i2):
    return tf.reduce_mean(tf.square(i1 - i2), axis=[1, 2, 3])  # use mean for normalised regulariser weighting


def gradient_dx(fv):
    return (fv[:, 2:, 1:-1] - fv[:, :-2, 1:-1]) / 2


def gradient_dy(fv):
    return (fv[:, 1:-1, 2:] - fv[:, 1:-1, :-2]) / 2


def gradient_txy(txy, fn):
    return tf.stack([fn(txy[..., i]) for i in [0, 1]], axis=3)


def gradient_norm(displacement, flag_l1=False):
    dtdx = gradient_txy(displacement, gradient_dx)
    dtdy = gradient_txy(displacement, gradient_dy)
    if flag_l1:
        norms = tf.abs(dtdx) + tf.abs(dtdy)
    else:
        norms = dtdx**2 + dtdy**2
    return tf.reduce_mean(norms, [1, 2, 3])


def bending_energy(displacement):
    dtdx = gradient_txy(displacement, gradient_dx)
    dtdy = gradient_txy(displacement, gradient_dy)
    dtdxx = gradient_txy(dtdx, gradient_dx)
    dtdyy = gradient_txy(dtdy, gradient_dy)
    dtdxy = gradient_txy(dtdx, gradient_dy)
    return tf.reduce_mean(dtdxx**2 + dtdyy**2 + 2*dtdxy**2, [1, 2, 3])
