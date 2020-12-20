# (Model) This can be implemented in MATLAB

import os
import matplotlib.pyplot as plt
import numpy as np


# data
data_dir = '../data'

# training
train_dir = '../data/train'
files_image_train = [fn for fn in os.listdir(train_dir) if os.path.splitext(fn)[0].isdigit()]
# save images to files and convert to standard orientation
images_train = np.stack([plt.imread(os.path.join(train_dir, fn))[::-1, ...].T for fn in files_image_train], axis=0)
# normalise individual images
images_train = (images_train-images_train.mean(axis=(1, 2), keepdims=True)) / images_train.std(axis=(1, 2), keepdims=True)
np.save(os.path.join(data_dir, 'images_train.npy'), images_train)

# test
test_dir = '../data/test'
files_image_test = [fn for fn in os.listdir(test_dir) if os.path.splitext(fn)[0].isdigit()]
files_label0_test = [os.path.splitext(fn)[0]+'_SPINAL_CORD'+os.path.splitext(fn)[1] for fn in files_image_test]
files_label1_test = [os.path.splitext(fn)[0]+'_BRAIN_STEM'+os.path.splitext(fn)[1] for fn in files_image_test]

images_test = np.stack([plt.imread(os.path.join(test_dir, fn))[::-1, ...].T for fn in files_image_test], axis=0)
images_test = (images_test-images_test.mean(axis=(1, 2), keepdims=True)) / images_test.std(axis=(1, 2), keepdims=True)
np.save(os.path.join(data_dir, 'images_test.npy'), images_test)

labels0_test = np.stack([plt.imread(os.path.join(test_dir, fn))[::-1, ...].T for fn in files_label0_test], axis=0)
labels1_test = np.stack([plt.imread(os.path.join(test_dir, fn))[::-1, ...].T for fn in files_label1_test], axis=0)

np.save(os.path.join(data_dir, 'labels_test.npy'), np.stack([labels0_test, labels1_test], axis=3))

# plot test data
for idx in range(len(files_image_test)):
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(images_test[idx,...], cmap='gray')
    ax.imshow(np.where(labels0_test[idx, ...] == 0, np.nan, labels0_test[idx, ...]), cmap='Reds_r')
    ax.imshow(np.where(labels1_test[idx, ...] == 0, np.nan, labels1_test[idx, ...]), cmap='Greens_r')
    ax.set_title(files_image_test[idx])
    plt.show()



