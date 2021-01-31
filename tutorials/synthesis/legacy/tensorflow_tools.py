import tensorflow as tf
import spatial_transformer_3d
# import conv3d_transpose


def linear(input, output_dim, scope=None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def gan_minibatch(input, w, b, num_kernels=100, kernel_dim=5):
    # x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    x = tf.matmul(input, w) + b
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], axis=1)


def nn_leaky_relu(x, alpha=0.2, name="leaky_relu"):
    with tf.variable_scope(name):
        # return tf.maximum(tf.minimum(0.0, alpha * x), x)
        return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)


def nn_resnet(x, w1, w2, strides, use_identity=True, use_lrelu=True, name="resnet"):
    # TODO: there is a bug in the variable_scope
    with tf.variable_scope(name):
        if use_lrelu:
            xr = nn_leaky_relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(x, w1, strides, "SAME")))
            xr2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(xr, w2, strides, "SAME"))
            if use_identity:
                x_out = nn_leaky_relu(xr2 + x)
            else:
                x_out = nn_leaky_relu(xr2)
        else:
            xr = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv2d(x, w1, strides, "SAME")))
            xr2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(xr, w2, strides, "SAME"))
            if use_identity:
                x_out = tf.nn.relu(xr2 + x)
            else:
                x_out = tf.nn.relu(xr2)
        return x_out

def l2_loss(x, batch_size):
    return tf.reduce_mean(tf.reshape(tf.mul(tf.square(x)), [batch_size, -1]))


def nn_conv3d_transpose(value, filters, output_shape, strides, padding="SAME", name=None):
    # return conv3d_transpose.conv3d_transpose(value, filters, output_shape, strides, padding=padding, name=name)
    return tf.nn.conv3d_transpose(value, filters, output_shape, strides, padding=padding, name=name)


def random_transform_vectors(batch_size):
    a = tf.random_normal([3, batch_size], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None)
    t = tf.random_normal([3, batch_size], mean=0.0, stddev=10.0, dtype=tf.float32, seed=None)
    s = tf.random_normal([3, batch_size], mean=1.0, stddev=0.2, dtype=tf.float32, seed=None)
    return a, t, s


def random_transform2(vol1, vol2, transform_vector, output_size):
    vol1 = spatial_transformer_3d.transformer(vol1, transform_vector, output_size, 'linear')
    vol2 = tf.to_int32(spatial_transformer_3d.transformer(vol2, transform_vector, output_size, 'nearest'))
    return vol1, vol2


def random_transform1(vol1, transform_vector, output_size):
    vol1 = spatial_transformer_3d.transformer(vol1, transform_vector, output_size, 'linear')
    return vol1


def random_transform3(vol1, vol2, vol3, transform_vector, output_size):
    vol1 = spatial_transformer_3d.transformer(vol1, transform_vector, output_size, 'linear')
    vol2 = tf.to_int32(spatial_transformer_3d.transformer(vol2, transform_vector, output_size, 'nearest'))
    vol3 = spatial_transformer_3d.transformer(vol3, transform_vector, output_size, 'linear')
    return vol1, vol2, vol3


def batch_norm3d(vols_batch):
    return [tf.contrib.layers.batch_norm(vols, center=True, scale=True, is_training=True, reuse=idx > 0,)
            for idx, vols in enumerate(vols_batch)]


def resize_volume(image, size, method=0, name='resizeVolume'):
    with tf.variable_scope(name):
        # size is [depth, height width]
        # image is Tensor with shape [batch, depth, height, width, channels]
        reshaped2D = tf.reshape(image,[-1, int(image.get_shape()[2]), int(image.get_shape()[3]), int(image.get_shape()[4])])
        resized2D = tf.image.resize_images(reshaped2D,[size[1],size[2]],method)
        reshaped3D = tf.reshape(resized2D,[int(image.get_shape()[0]), int(image.get_shape()[1]), size[1], size[2], int(image.get_shape()[4])])
        permuted = tf.transpose(reshaped3D,[0,3,2,1,4])
        reshaped2DB = tf.reshape(permuted,[-1, size[1], int(image.get_shape()[1]), int(image.get_shape()[4])])
        resized2DB = tf.image.resize_images(reshaped2DB,[size[1],size[0]],method)
        reshaped3DB = tf.reshape(resized2DB,[int(image.get_shape()[0]), size[2], size[1], size[0], int(image.get_shape()[4])])
        return tf.transpose(reshaped3DB, [0, 3, 2, 1, 4])


def resize_grid(grid, size):
    return tf.stack(tf.image.resize_bilinear(tf.unstack(grid, axis=3), size), axis=3)