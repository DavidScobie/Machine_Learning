# (Model) This uses TensorFlow

import tensorflow as tf
import networks
import numpy as np
from scipy.io import savemat


# 0 - read all the data
data_dir = '../data/'
images = np.load(data_dir+'images_test.npy')
image_size = [images.shape[1], images.shape[2]]
labels = np.load(data_dir+'labels_test.npy')

# model_dir = data_dir
# model_dir = '../trained/ncc_gn_1e-2/'
# model_dir = '../trained/ncc_gn_1e-1/'
# model_dir = '../trained/ncc_gn_1e0/'
model_dir = '../trained/ncc_gn_1e1/'

file_model_save = model_dir+'model_saved'

# for testing - the No.2 image is for testing
indices_moving = [0, 0, 1, 1, 2, 2]
indices_fixed = [1, 2, 0, 2, 0, 1]
num_data = len(indices_moving)

# 2 - graph
# network for predicting ddf only
ph_moving_image = tf.placeholder(tf.float32, [num_data]+image_size)
ph_fixed_image = tf.placeholder(tf.float32, [num_data]+image_size)
input_moving_image = tf.expand_dims(ph_moving_image, axis=3)  # data augmentation
input_fixed_image = tf.expand_dims(ph_fixed_image, axis=3)  # data augmentation

reg_net = networks.RegNet2D(minibatch_size=num_data, image_moving=input_moving_image, image_fixed=input_fixed_image)

# restore the trained weights
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, file_model_save)


# 3 - inference
testFeed = {ph_moving_image: images[indices_moving, ...], ph_fixed_image: images[indices_fixed, ...]}

# compute ddf
ddfs, resampling_grids = sess.run([reg_net.ddf, reg_net.grid_warped], feed_dict=testFeed)

# warp the test images and labels  (by building a new mini computation graph, amongst alternative methods)
ph_moving = tf.placeholder(tf.float32, [num_data]+image_size)
warped = tf.contrib.resampler.resampler(tf.expand_dims(ph_moving, axis=3), resampling_grids)  # new node
with tf.Session() as sess:
    warped_images = sess.run(warped, feed_dict={ph_moving: images[indices_moving, ...]})
    warped_labels0 = sess.run(warped, feed_dict={ph_moving: labels[indices_moving, ..., 0]})
    warped_labels1 = sess.run(warped, feed_dict={ph_moving: labels[indices_moving, ..., 1]})

# save to files for validation
np.save(model_dir+'indices_moving.npy', indices_moving)
np.save(model_dir+'indices_fixed.npy', indices_fixed)
np.save(model_dir+'ddfs.npy', ddfs)
np.save(model_dir+'resampling_grids.npy', resampling_grids)
np.save(model_dir+'warped_images.npy', warped_images)
np.save(model_dir+'warped_labels0.npy', warped_labels0)
np.save(model_dir+'warped_labels1.npy', warped_labels1)

savemat(model_dir+'reg',
        {'indices_moving': indices_moving,
         'indices_fixed': indices_fixed,
         'ddfs': ddfs,
         'resampling_grids': resampling_grids,
         'warped_images': warped_images,
         'warped_labels0': warped_labels0,
         'warped_labels1': warped_labels1})
