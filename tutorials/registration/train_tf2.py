# (Model) this uses TensorFlow-2
import random
import os

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

import tf_utils as utils


## read all the data
PATH_TO_TRAIN = 'data/datasets-hn2dct/train'
PATH_TO_TEST = 'data/datasets-hn2dct/test'
PATH_TO_RESULT = 'result'

images = np.stack([mpimg.imread(os.path.join(PATH_TO_TRAIN, f)) for f in os.listdir(PATH_TO_TRAIN) if f.endswith('.png')],axis=0)  # stack at dim=0 consistent with tf
images = np.pad(images, [(0,0),(0,0),(0,1)])  # padding for an easier image size
image_size = (images.shape[1], images.shape[2])
num_data = images.shape[0]


test_images = np.stack([mpimg.imread(os.path.join(PATH_TO_TEST, f)) for f in os.listdir(PATH_TO_TEST) if (f.find('_')==-1 and f.endswith('.png'))],axis=0) 
test_images = np.pad(images, [(0,0),(0,0),(0,1)])  # padding for an easier image size

## settings
weight_regulariser = 1e-1
minibatch_size = 8
learning_rate = 1e-3
total_iterations = int(1e5)
freq_info_print = 500
freq_test_save = 5000


## network
reg_net = utils.UNet(out_channels=2, num_channels_initial=16)  # output ddfs in x,y two channels
reg_net = reg_net.build(input_shape=image_size+(2,))
optimizer = tf.optimizers.Adam(learning_rate)

## train step
@tf.function
def train_step(mov_images, fix_images):
    with tf.GradientTape() as tape:
        inputs = tf.stack([mov_images,fix_images],axis=3)
        ddfs = reg_net(inputs, training=True)
        pre_images = utils.warp_images(mov_images,ddfs)
        loss_similarity = tf.reduce_mean(utils.sum_square_difference(fix_images, pre_images))
        loss_regularise = tf.reduce_mean(utils.bending_energy(ddfs))
        loss = loss_similarity + loss_regularise*weight_regulariser
    gradients = tape.gradient(loss, reg_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, reg_net.trainable_variables))
    return loss, loss_similarity, loss_regularise

## test
def test_step(mov_images, fix_images):
    inputs = tf.stack([mov_images,fix_images],axis=3)
    ddfs = reg_net(inputs, training=False)
    pre_images = utils.warp_images(mov_images,ddfs)
    loss_similarity = tf.reduce_mean(utils.sum_square_difference(fix_images, pre_images))
    loss_regularise = tf.reduce_mean(utils.bending_energy(ddfs))
    loss = loss_similarity + loss_regularise*weight_regulariser
    return loss, loss_similarity, loss_regularise, pre_images


## training
num_minibatch = int(num_data/minibatch_size/2)
train_indices = [i for i in range(num_data)]
test_indices = [[0,0,1,1,2,2],[1,2,0,2,0,1]] # [moving,fixed]
for step in range(total_iterations):

    if step in range(0, total_iterations, num_minibatch):
        random.shuffle(train_indices)

    minibatch_idx = step % num_minibatch
    # random pairs
    indices_moving = train_indices[minibatch_idx*minibatch_size:(minibatch_idx+1)*minibatch_size]
    indices_fixed = train_indices[::-1][minibatch_idx*minibatch_size:(minibatch_idx+1)*minibatch_size]

    loss_train, loss_sim_train, loss_reg_train = train_step(images[indices_moving,...],images[indices_fixed,...])

    if step in range(0, total_iterations, freq_info_print):
        print('Step %d: Loss=%f (similarity=%f, regulariser=%f)' %
              (step, loss_train, loss_sim_train, loss_reg_train))
        print('  Moving-fixed image pair indices: %s - %s' % (indices_moving, indices_fixed))
    
    if step in range(0, total_iterations, freq_test_save):
        loss_test, loss_sim_test, loss_reg_test, pre_images_test = test_step(images[test_indices[0],...],images[test_indices[1],...])
        print('*** Test *** Step %d: Loss=%f (similarity=%f, regulariser=%f)' %
              (step, loss_test, loss_sim_test, loss_reg_test))
        filepath_to_save = os.path.join(PATH_TO_RESULT, "test_step%06d-tf.npy" % step)
        np.save(filepath_to_save, pre_images_test)
        tf.print('Test data saved: {}'.format(filepath_to_save))

print('Training done.')
        