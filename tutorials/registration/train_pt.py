# This script uses PyTorch
import random
import os

import torch
import numpy as np

import pt_utils as utils
from np_utils import get_image_arrays


os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
PATH_TO_RESULT = 'result'


## read all the data
images, test_images, test_indices = get_image_arrays()
image_size = (images.shape[1], images.shape[2])
num_data = images.shape[0]

## settings
weight_regulariser = 0.01
minibatch_size = 8
learning_rate = 1e-3
total_iterations = int(2e4+1)
freq_info_print = 200
freq_test_save = 2000

## network
reg_net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=2, out_channels=2, init_features=32, pretrained=False)
if use_cuda:
    reg_net.cuda()

## training
num_minibatch = int(num_data/minibatch_size/2)
train_indices = [i for i in range(num_data)]
# optimisation loop
for step in range(total_iterations):

    if step in range(0, total_iterations, num_minibatch):
        random.shuffle(train_indices)
    
    minibatch_idx = step % num_minibatch
    # random pairs
    indices_moving = train_indices[minibatch_idx*minibatch_size:(minibatch_idx+1)*minibatch_size]
    indices_fixed = train_indices[::-1][minibatch_idx*minibatch_size:(minibatch_idx+1)*minibatch_size]

    moving_images = torch.tensor(images[indices_moving,...])
    fixed_images = torch.tensor(images[indices_fixed,...])
    if use_cuda:
        moving_images, fixed_images = moving_images.cuda(), fixed_images.cuda()

    optimizer.zero_grad()
    ddfs = reg_net(torch.stack((moving_images,fixed_images),dim=1))
    pre_images  = utils.warp_images(moving_images, ddfs)
    loss_sim_train = utils.square_difference(pre_images, fixed_images)
    loss_reg_train = utils.gradient_norm(ddfs)
    loss_train = loss_sim_train + loss_reg_train*weight_regulariser
    loss_train.backward()
    optimizer.step()

    # Compute and print loss
    if step in range(0, total_iterations, freq_info_print):
        print('Step %d: Loss=%f (similarity=%f, regulariser=%f)' % (step, loss_train, loss_sim_train, loss_reg_train))
        print('  Moving-fixed image pair indices: %s - %s' % (indices_moving, indices_fixed))

    # --- testing during training (no validation labels available)
    if step in range(0, total_iterations, freq_test_save):
        moving_images_test = torch.tensor(test_images[test_indices[0],...])
        fixed_images_test = torch.tensor(test_images[test_indices[1],...])
        if use_cuda:
            moving_images_test, fixed_images_test = moving_images_test.cuda(), fixed_images_test.cuda()
        
        ddfs_test = reg_net(torch.stack((moving_images_test,fixed_images_test),dim=1))
        pre_images_test  = utils.warp_images(moving_images_test, ddfs_test)
        loss_sim_test = utils.square_difference(pre_images, fixed_images_test)
        loss_reg_test = utils.gradient_norm(ddfs_test)
        loss_test = loss_sim_test + loss_reg_test*weight_regulariser

        print('*** Test *** Step %d: Loss=%f (similarity=%f, regulariser=%f)' % (step, loss_test, loss_sim_test, loss_reg_test))
        filepath_to_save = os.path.join(PATH_TO_RESULT, "test_step%06d-tf.npy" % step)
        np.save(filepath_to_save, pre_images_test)
        tf.print('Test data saved: {}'.format(filepath_to_save))

print('Training done.')

        