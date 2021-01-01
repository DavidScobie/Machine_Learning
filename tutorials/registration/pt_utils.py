import torch


## transformation functions
def get_reference_grid(grid_size):
    # grid_size: [batch_size, height, width]
    grid = tf.cast(tf.stack(tf.meshgrid(
                        tf.range(grid_size[1]),
                        tf.range(grid_size[2]),
                        indexing='ij'), axis=2), dtype=tf.float32)
    return tf.tile(tf.expand_dims(grid, axis=0), [grid_size[0],1,1,1])

def warp_images(images, ddfs):
    # images: [batch_size, height, width]
    # ddfs: [batch_size, height, width, 2]
    reference_grid = get_reference_grid(ddfs.shape[0:3])
    warped_grids = reference_grid + ddfs
    return bilinear_resampler(images, warped_grids)

## loss functions
def square_difference(i1, i2):
    return tf.reduce_mean(tf.square(i1 - i2), axis=[1, 2])  # use mean for normalised regulariser weighting


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