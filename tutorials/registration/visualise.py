# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
# run train.py before visualise the results
import os

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


PATH_TO_TEST = 'data/datasets-hn2dct/test'
PATH_TO_RESULT = 'result'
test_indices = [[0,0,1,1,2,2],[1,2,0,2,0,1]]
test_images = np.stack([mpimg.imread(os.path.join(PATH_TO_TEST, f)) for f in os.listdir(PATH_TO_TEST) if (f.find('_')==-1 and f.endswith('.png'))],axis=0) 
test_images = np.pad(test_images, [(0,0),(0,0),(0,1)])  # padding for an easier image size

# to plot example slices of segmentation results
for ext in ["-tf.npy","-pt.npy"]:  # find all npy files
    files = [f for f in os.listdir(PATH_TO_RESULT) if f.endswith(ext)]
    if len(files)==0: continue
    pre_images = np.load(os.path.join(PATH_TO_RESULT,max(files)))  # find the maximum step

    for ii in range(pre_images.shape[0]):
        plt.figure()
        axs = plt.subplot(1, 3, 1)
        axs.set_title('moving')
        axs.imshow(test_images[test_indices[0][ii],...], cmap='gray')
        axs.axis('off')

        axs = plt.subplot(1, 3, 2)
        axs.set_title('registered')
        axs.imshow(pre_images[ii,...], cmap='gray')
        axs.axis('off')

        axs = plt.subplot(1, 3, 3)
        axs.set_title('fixed')
        axs.imshow(test_images[test_indices[1][ii],...], cmap='gray')
        axs.axis('off')

        # plt.show()
        plt.savefig(os.path.join(PATH_TO_RESULT, '{}-{}.jpg'.format(max(files).split('.')[0],ii)))
        plt.close()

print('Plots saved: {}'.format(os.path.abspath(PATH_TO_RESULT)))
