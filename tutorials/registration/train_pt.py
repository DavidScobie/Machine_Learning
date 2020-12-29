# (Model) this uses TensorFlow-2
import random
import os

import torch
import numpy as np

import pt_utils as utils
from np_utils import get_image_arrays


os.environ["CUDA_VISIBLE_DEVICES"]="2"
PATH_TO_RESULT = 'result'


## read all the data
images, test_images, test_indices = get_image_arrays()
image_size = (images.shape[1], images.shape[2])
num_data = images.shape[0]


## settings
weight_regulariser = 0.01
minibatch_size = 8
learning_rate = 1e-4
total_iterations = int(3e5+1)
freq_info_print = 500
freq_test_save = 5000


## network
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=2, out_channels=2, init_features=16, pretrained=False)
if use_cuda:
    model.cuda()

## training

# optimisation loop
freq_print = 100  # in steps
freq_test = 2000  # in steps
total_steps = int(2e5)
step = 0
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
while step < total_steps:
    for ii, (images, labels) in enumerate(train_loader):
        step += 1
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        ddfs = model(images)
        pre_images  = utils.warp_images(moving_images)
        loss_dissimilarity = utils.square_difference(pre_images, fixed_images)
        loss_regualrisation = utils.radient_norm(ddfs)
        loss = loss_dissimilarity + loss_regualrisation*weight_regulariser
        loss.backward()
        optimizer.step()

        # Compute and print loss
        if (step % freq_print) == 0:    # print every freq_print mini-batches
            print('Step %d loss: %.5f' % (step,loss.item()))

        # --- testing during training (no validation labels available)
        if (step % freq_test) == 0:  
            images_test, id_test = iter(test_loader).next()  # test one mini-batch
            if use_cuda:
                images_test = images_test.cuda()
            preds_test = model(images_test)
            for idx, index in enumerate(id_test):
                filepath_to_save = os.path.join(RESULT_PATH,"label_test%02d_step%06d-pt.npy" % (index,step))
                np.save(filepath_to_save, preds_test.detach()[idx,...].cpu().numpy().squeeze())
                print('Test data saved: {}'.format(filepath_to_save))

print('Training done.')

        