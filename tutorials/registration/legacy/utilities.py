# (Model) This can be implemented in MATLAB

import numpy as np
# from scipy.ndimage import morphology


def compute_dice_batch(a, b):
    a = a > .5
    b = b > .5
    return 2*np.sum(a*b, axis=(1, 2)) / (np.sum(a, axis=(1, 2))+np.sum(b, axis=(1, 2)))


def compute_dice(a, b):
    a = a > .5
    b = b > .5
    return 2*np.sum(a*b) / (np.sum(a)+np.sum(b))


def hausdorff_distance(a, b, percentile=95, sampling=1):
    sa = extract_surface(a)
    sb = extract_surface(b)
    dta = morphology.distance_transform_edt(~sa, sampling)
    dtb = morphology.distance_transform_edt(~sb, sampling)
    sds = np.concatenate([np.ravel(dta[sb != 0]), np.ravel(dtb[sa != 0])])
    # msd = np.mean(sds)
    # rms = np.sqrt((sds**2).mean())
    # hd  = np.max(sds)
    return np.percentile(sds, percentile)


'''
def extract_surface(a, connectivity=1):
    a = np.atleast_1d(a.astype(np.bool))

    if a.ndim == 3:  # 3D
        conn = np.zeros((3, 3, 3), dtype=bool)
        # only considering in-slice voxels as surface
        conn[:, :, 1] = morphology.generate_binary_structure(2, connectivity )

    else:  # 2D
        conn = morphology.generate_binary_structure(a.ndim, connectivity)

    return np.bitwise_xor(a, morphology.binary_erosion(a, conn))
'''
