# (Model) this uses TensorFlow-2

import random
import time
import sys
import os

import tensorflow as tf
import numpy as np

import networks_tf2 as network
import losses_tf2 as loss


# 0 - read all the data
data_dir = '../data/'
images = np.load(data_dir+'images_train.npy')
image_size = [images.shape[1], images.shape[2]]
num_data = images.shape[0]


# 1 - settings
weight_regulariser = 1e-1

minibatch_size = 8
learning_rate = 1e-3
total_iterations = int(50000+1)
freq_info_print = 200
freq_model_save = 5000

model_dir_save = data_dir + 'arg_%s/' % weight_regulariser
if not os.path.exists(model_dir_save):
        os.makedirs(model_dir_save)
model_file_save = model_dir_save + 'model_saved'


# 2 - graph
ph_moving_image = tf.placeholder(tf.float32, [minibatch_size]+image_size)
ph_fixed_image = tf.placeholder(tf.float32, [minibatch_size]+image_size)
input_moving_image = tf.expand_dims(ph_moving_image, axis=3)  # data augmentation here
input_fixed_image = tf.expand_dims(ph_fixed_image, axis=3)  # data augmentation here

# instantiate a network
reg_net = networks.RegNet2D(minibatch_size=minibatch_size, image_moving=input_moving_image, image_fixed=input_fixed_image)

# loss
warped_moving_image = reg_net.warp_image(input_moving_image)  # warp the moving image with the predicted ddf

loss_similarity = tf.reduce_mean(loss2.normalised_cross_correlation(warped_moving_image, input_fixed_image))
# loss_similarity = tf.reduce_mean(loss2.sum_square_difference(warped_moving_image, input_fixed_image))
loss_regularise = tf.reduce_mean(loss.bending_energy(reg_net.ddf))

loss = loss_similarity + weight_regulariser*loss_regularise
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# 3 - training
num_minibatch = int(num_data/minibatch_size/2)
train_indices = [i for i in range(num_data)]

saver = tf.train.Saver(max_to_keep=1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(total_iterations):

    if step in range(0, total_iterations, num_minibatch):
        random.shuffle(train_indices)

    minibatch_idx = step % num_minibatch
    # random pairs
    indices_moving = train_indices[minibatch_idx*minibatch_size:(minibatch_idx+1)*minibatch_size]
    indices_fixed = train_indices[::-1][minibatch_idx*minibatch_size:(minibatch_idx+1)*minibatch_size]
    trainFeed = {ph_moving_image: images[indices_moving, ...], ph_fixed_image: images[indices_fixed, ...]}

    sess.run(train_op, feed_dict=trainFeed)

    if step in range(0, total_iterations, freq_info_print):
        current_time = time.asctime(time.gmtime())
        loss_train, loss_sim_train, loss_reg_train = sess.run(
            [loss, loss_similarity, loss_regularise], feed_dict=trainFeed)

        print('Step %d [%s]: Loss=%f (similarity=%f, regulariser=%f)' %
              (step, current_time, loss_train, loss_sim_train, loss_reg_train))
        print('  Moving-fixed image pair indices: %s - %s' % (indices_moving, indices_fixed))

    if step in range(0, total_iterations, freq_model_save):
        save_path = saver.save(sess, model_file_save, write_meta_graph=False)
        print("Model saved in: %s" % save_path)
