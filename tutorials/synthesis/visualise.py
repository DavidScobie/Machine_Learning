# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
# run train_*.py before visualise the results
import os

import numpy as np
import matplotlib.pyplot as plt


PATH_TO_RESULT = 'result'

# to plot example slices of segmentation results
for ext in ["-tf.npy","-pt.npy"]:  # find all npy files
    files = [f for f in sorted(os.listdir(PATH_TO_RESULT)) if f.endswith(ext)]
    if len(files)==0: continue
    for ii, filename in enumerate(files):
        images = np.load(os.path.join(PATH_TO_RESULT,filename))
        images = np.reshape(images,(-1,images.shape[1]*8,images.shape[2]))
        images = np.reshape(np.transpose(images,[0,2,1]),(-1,images.shape[1]))

        plt.figure()
        plt.imshow(images,cmap='gray')
        # plt.show()
        plt.savefig(os.path.join(PATH_TO_RESULT, '{}-{}.jpg'.format(filename.split('.')[0],ii)))
        plt.close()

print('Plots saved: {}'.format(os.path.abspath(PATH_TO_RESULT)))
