# (Model) This uses TensorFlow

import tensorflow as tf


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
