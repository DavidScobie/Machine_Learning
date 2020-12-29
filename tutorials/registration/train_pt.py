# (Model) this uses TensorFlow-2
import random
import os

import torch
import numpy as np

import tf_utils as utils
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


## training

        