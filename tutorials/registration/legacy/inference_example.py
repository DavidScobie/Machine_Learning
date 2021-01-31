# (Provided) This uses TensorFlow
# *** Available as part of UCL MPHY0025 (Information Processing in Medical Imaging) Assessed Coursework 2018-19 ***
# *** This code is with an Apache 2.0 license, University College London ***

import tensorflow as tf
import networks
import numpy as np
from matplotlib.pyplot import imread


# 1 - Read images and convert to "standard orientation"
files_image_test = ['../data/test/433.png', '../data/test/441.png']
images = np.stack([imread(fn)[::-1, ...].T for fn in files_image_test], axis=0)

# Normalise the test images so they have zero-mean and unit-variance
images = (images-images.mean(axis=(1, 2), keepdims=True)) / images.std(axis=(1, 2), keepdims=True)
image_size = [images.shape[1], images.shape[2]]

# 2 - Load one of the provided trained networks
model_dir = '../trained/ssd_1e1/'
# model_dir = '../trained/ssd_1e0/'
# model_dir = '../trained/ssd_1e-1/'
# model_dir = '../trained/ssd_1e-2/'
file_model_save = model_dir+'model_saved'

# Restore the computation graph
ph_moving_image = tf.placeholder(tf.float32, [1]+image_size)
ph_fixed_image = tf.placeholder(tf.float32, [1]+image_size)
input_moving_image = tf.expand_dims(ph_moving_image, axis=3)
input_fixed_image = tf.expand_dims(ph_fixed_image, axis=3)
reg_net = networks.RegNet2D(minibatch_size=1, image_moving=input_moving_image, image_fixed=input_fixed_image)

# Reinstate the trained weights (stored in the network model file)
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, file_model_save)


# 5 - Predict the displacement field
testFeed = {ph_moving_image: images[[0], ...], ph_fixed_image: images[[1], ...]}
ddfs, resampling_grids = sess.run([reg_net.ddf, reg_net.grid_warped], feed_dict=testFeed)


# 6a - an example if one uses NumPy for analysis
"""
np.save(model_dir+'ddfs.npy', ddfs)
np.save(model_dir+'resampling_grids.npy', resampling_grids)
"""

# 6b - an example if one uses MATLAB for analysis
"""
from scipy.io import savemat
savemat(model_dir+'reg',
        {'ddfs': ddfs,
         'resampling_grids': resampling_grids})
"""
