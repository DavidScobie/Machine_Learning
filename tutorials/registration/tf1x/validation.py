# (Model) This can be implemented in MATLAB

import utilities
import numpy as np
import matplotlib.pyplot as plt


data_dir = '../data/'
images = np.load(data_dir+'images_test.npy')
image_size = [images.shape[1], images.shape[2]]
labels = np.load(data_dir+'labels_test.npy')

# load data from inference output
model_dir = '../trained/ncc_gn_1e1/'
ddfs = np.load(model_dir+'ddfs.npy')
indices_moving = np.load(model_dir+'indices_moving.npy')
indices_fixed = np.load(model_dir+'indices_fixed.npy')
warped_images = np.load(model_dir+'warped_images.npy')
warped_labels0 = np.load(model_dir+'warped_labels0.npy')
warped_labels1 = np.load(model_dir+'warped_labels1.npy')
num_data = len(indices_moving)

# numerical results
for idx in range(num_data):
    print('Case%d (Moving:%d - Fixed:%d) ' % (idx, indices_moving[idx], indices_fixed[idx]))
    # Dice
    print('Label0: Dice before: %f'
          % utilities.compute_dice(labels[indices_fixed[idx], ..., 0], labels[indices_moving[idx], ..., 0]))
    print('Label0: Dice after: %f'
          % utilities.compute_dice(labels[indices_fixed[idx], ..., 0], warped_labels0[idx, ..., -1]))
    print('Label1: Dice before: %f'
          % utilities.compute_dice(labels[indices_fixed[idx], ..., 1], labels[indices_moving[idx], ..., 1]))
    print('Label1: Dice after: %f'
          % utilities.compute_dice(labels[indices_fixed[idx], ..., 1], warped_labels1[idx, ..., -1]))
    # Boundary distance
    print('Label0: 95th-Hausdorff before: %f'
          % utilities.hausdorff_distance(labels[indices_fixed[idx], ..., 0], labels[indices_moving[idx], ..., 0], 95))
    print('Label0: 95th-Hausdorff after: %f'
          % utilities.hausdorff_distance(labels[indices_fixed[idx], ..., 0], warped_labels0[idx, ..., -1], 95))
    print('Label1: 95th-Hausdorff before: %f'
          % utilities.hausdorff_distance(labels[indices_fixed[idx], ..., 1], labels[indices_moving[idx], ..., 1], 95))
    print('Label1: 95th-Hausdorff after: %f'
          % utilities.hausdorff_distance(labels[indices_fixed[idx], ..., 1], warped_labels1[idx, ..., -1], 95))


# plot
for idx in range(num_data):
    plt.figure()
    plt.subplot(1, 3, 1), plt.imshow(images[indices_moving[idx], ...].T, origin='lower', cmap='gray'), plt.title('Moving Image')
    plt.subplot(1, 3, 2), plt.imshow(images[indices_fixed[idx], ...].T, origin='lower', cmap='gray'), plt.title('Fixed Image')
    plt.subplot(1, 3, 3), plt.imshow(warped_images[idx, ..., -1].T, origin='lower', cmap='gray'), plt.title('Warped Image')

    plt.figure()  # original diff, current diff; ddf-x, ddf-y
    plt.subplot(2, 2, 1), plt.imshow((images[indices_moving[idx], ...]-images[indices_fixed[idx], ...]).T, origin='lower', cmap='gray'), plt.title('Moving - Fixed')
    plt.subplot(2, 2, 2), plt.imshow((warped_images[idx, ..., -1]-images[indices_fixed[idx], ...]).T, origin='lower', cmap='gray'), plt.title('Warped - Fixed')
    plt.subplot(2, 2, 3), plt.imshow(ddfs[idx, ..., 1].T, origin='lower', cmap='gray'), plt.title('DX')
    plt.subplot(2, 2, 4), plt.imshow(ddfs[idx, ..., 0].T, origin='lower', cmap='gray'), plt.title('DY')

    plt.figure()
    plt.subplot(2, 3, 1), plt.imshow(labels[indices_moving[idx], ..., 0].T, origin='lower', cmap='gray'), plt.title('Moving Label')
    plt.subplot(2, 3, 2), plt.imshow(labels[indices_fixed[idx], ..., 0].T, origin='lower', cmap='gray'), plt.title('Fixed Label')
    plt.subplot(2, 3, 3), plt.imshow(warped_labels0[idx, ..., -1].T, origin='lower', cmap='gray'), plt.title('Warped Label')
    plt.subplot(2, 3, 4), plt.imshow(labels[indices_moving[idx], ..., 1].T, origin='lower', cmap='gray')
    plt.subplot(2, 3, 5), plt.imshow(labels[indices_fixed[idx], ..., 1].T, origin='lower', cmap='gray')
    plt.subplot(2, 3, 6), plt.imshow(warped_labels1[idx, ..., -1].T, origin='lower', cmap='gray')


