
import numpy as np
import h5py
import os


# ---------- handling data ----------
class SimpleFrameFeeder:
    def __init__(self, filename):
        self.filename = filename
        self.file_id = h5py.File(self.filename, 'r')  # for single file format

    def get_batch(self, data_indices):
        # for serialised data
        group_names = ['/frame%06d' % i for i in data_indices]
        return np.concatenate([np.expand_dims(np.expand_dims(self.file_id[i], axis=0), axis=3) for i in group_names], axis=0)


class TrackedFrameFeeder:
    def __init__(self, filename_img, filename_cond, random_cond=False, is_grid=False):
        self.filename_img = filename_img
        self.file_id_img = h5py.File(self.filename_img, 'r')  # for single file format
        self.filename_cond = filename_cond
        self.file_id_cond = h5py.File(self.filename_cond, 'r')  # for single file format
        self.random_cond = random_cond
        self.is_grid = is_grid

    def get_batch(self, data_indices):
        # for serialised data
        img = self.get_batch_img_only(data_indices)
        if self.is_grid:
            cond = self.get_batch_grid_only(data_indices)
        else:
            cond = self.get_batch_tfm_only(data_indices)
        return img, cond

    def get_batch_img_only(self, data_indices):
        # for serialised data
        group_names = ['/frame%06d' % i for i in data_indices]
        return np.concatenate([np.expand_dims(np.expand_dims(self.file_id_img[i], axis=0), axis=3) for i in group_names], axis=0)

    def get_batch_tfm_only(self, data_indices):
        # for serialised data
        group_names = ['/frame%06d' % i for i in data_indices]
        output_tfm = np.concatenate([np.reshape(self.file_id_cond[i], [1, -1]) for i in group_names], axis=0)
        if self.random_cond:
            output_tfm[:, 9:12] = output_tfm[:, 9:12] + np.random.normal(loc=0, scale=0.1, size=[len(data_indices), 3])  # position only
        return output_tfm

    def get_batch_grid_only(self, data_indices):
        # for serialised data
        group_names = ['/frame%06d' % i for i in data_indices]
        output_grid = np.concatenate([np.expand_dims(np.transpose(self.file_id_cond[i], [1, 2, 0]), axis=0) for i in group_names], axis=0)
        return output_grid


class DataFeeder:
    def __init__(self, filename, case_size=0):
        self.filename = filename
        self.file_id = h5py.File(self.filename, 'r')  # for single file format
        self.case_size = case_size

    def get_batch(self, data_indices):
        # for serialised data
        group_names = ['/ss%06d' % i for i in data_indices]
        return np.concatenate([np.expand_dims(self.file_id[i], axis=0) for i in group_names], axis=0)

    def get_cases(self, case_indices):
        # case_indices = [int(x/self.case_size) for x in data_indices]  # for serialised data
        group_names = ['/case%04d' % i for i in case_indices]
        return np.concatenate([np.expand_dims(np.expand_dims(self.file_id[i], axis=4), axis=0) for i in group_names], axis=0)


class ClusterDataFeeder:
    def __init__(self, filename):
        self.filename = filename
        self.file_id = h5py.File(self.filename, 'r')  # for single file format
        self.case_indices = list(self.file_id['/caseIndices'].value.squeeze())
        self.cluster_size = self.file_id['/clusterSize'][0]
        self.num_cases = len(set(self.case_indices))

    def get_batch(self, data_indices):
        # for serialised data
        group_names = ['/ss%06d' % i for i in data_indices]
        return np.concatenate([np.expand_dims(self.file_id[i], axis=0) for i in group_names], axis=0)

    def get_case_indices(self, data_indices):
        # for serialised data
        case_indices = [self.case_indices[i] for i in data_indices]
        return case_indices


class NoiseFeeder:
    def __init__(self, dim, batch_size, p1=0, p2=1):
        self.p1 = p1
        self.p2 = p2
        self.dim = dim
        self.batch_size = batch_size

    def get_batch(self):
        # return np.random.uniform(low=self.p1, high=self.p2, size=[self.batch_size, self.dim])
        return np.random.normal(loc=self.p1, scale=self.p2, size=[self.batch_size, self.dim])
        # N.B. this collapses
        # return np.sort(np.random.uniform(low=self.p1, high=self.p2, size=[self.batch_size, self.dim]), axis=1)

    def get_batch_linspace_samples(self, idx_dim, bounds=3):
        noise_batch = np.ones(shape=[self.batch_size, self.dim])*self.p1
        noise_batch[:, idx_dim] = np.linspace(-bounds*self.p2, bounds*self.p2, self.batch_size)+self.p1
        return noise_batch


def random_transform_generator(batch_size, cornerScale=.1):

    offsets = np.tile([[[1.,1.,1.],[1.,1.,-1.],[1.,-1.,1.],[-1.,1.,1.]]],[batch_size,1,1])*np.random.uniform(0,cornerScale,[batch_size,4,3])
    newCorners = np.transpose(np.concatenate((np.tile([[[-1.,-1.,-1.],[-1.,-1.,1.],[-1.,1.,-1.],[1.,-1.,-1.]]],[batch_size,1,1])+offsets,np.ones([batch_size,4,1])),2),[0,1,2]) # O = T I
    srcCorners=np.tile(np.transpose([[[-1.,-1.,-1.,1.],[-1.,-1.,1.,1.],[-1.,1.,-1.,1.],[1.,-1.,-1.,1.]]],[0,1,2]),[batch_size,1,1])
    transforms = np.array([np.linalg.lstsq(srcCorners[k], newCorners[k])[0] for k in range(srcCorners.shape[0])])
    transforms = transforms*np.concatenate((np.ones([batch_size,1,2]),(-1)**np.random.randint(0,2,[batch_size,1,1]),np.ones([batch_size,1,1])),2) # random LR flip
    transforms = np.reshape(np.transpose(transforms[:][:,:][:,:,:3],[0,2,1]),[-1,1,12])
    return transforms


def initial_transform_generator(batch_size):

    identity = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
    identity = identity.flatten()
    transforms = np.reshape(np.tile(identity,batch_size),[batch_size,1,12])
    return transforms


def dataset_switcher(dataset_name='skull'):

    if dataset_name[0:3] == 'all':
        filename = 'Scratch/data/protocol/normalised/all_96x128.h5'
        data_size = [128, 96]
    elif dataset_name[0:3] == 'sku':
        filename = 'Scratch/data/protocol/normalised/skull_96x128.h5'
        data_size = [128, 96]
    elif dataset_name[0:3] == 'abd':
        filename = 'Scratch/data/protocol/normalised/abdomen_96x128.h5'
        data_size = [128, 96]
    elif dataset_name[0:3] == 'sim':
        # filename = 'Scratch/data/fetusphan/normalised/images_120x160_norm.h5'
        filename = 'Scratch/data/fetusphan/normalised/images1h_120x160_norm.h5'  # 82538
        data_size = [160, 120]
    elif dataset_name[0:3] == 'img':
        filename = 'Scratch/data/fetusphan/normalised/images0_120x160_norm.h5'
        data_size = [160, 120]
    elif dataset_name[0:3] == 'im1':
        filename = 'Scratch/data/fetusphan/normalised/images0_240x320_norm.h5'
        data_size = [320, 240]
    elif dataset_name[0:3] == 'im0':
        filename = 'Scratch/data/fetusphan/normalised/images0_480x640_norm.h5'
        data_size = [640, 480]
    elif dataset_name[0:3] == 'tfm':
        # filename = 'Scratch/data/fetusphan/normalised/labels_3x4_tform.h5'
        filename = 'Scratch/data/fetusphan/normalised/labels1h_3x4_tform.h5'
        data_size = 12
    elif dataset_name[0:3] == 'pos':
        # filename = 'Scratch/data/fetusphan/normalised/labels_3x1_pos.h5'
        filename = 'Scratch/data/fetusphan/normalised/labels1h_3x1_pos.h5'
        data_size = 3
    elif dataset_name[0:3] == 'grd':
        filename = 'Scratch/data/fetusphan/normalised/labels0_120x160x3_grid_norm.h5'
        data_size = [3, 160, 120]
    elif dataset_name[0:3] == 'gr1':
        filename = 'Scratch/data/fetusphan/normalised/labels0_240x320x3_grid_norm.h5'
        data_size = [3, 320, 240]
    elif dataset_name[0:3] == 'gr0':
        filename = 'Scratch/data/fetusphan/normalised/labels0_480x640x3_grid_norm.h5'
        data_size = [3, 640, 480]
    elif dataset_name[0:3] == 'roi':
        filename = 'Scratch/data/fetusphan/normalised/groups0_5_indices.h5'
        data_size = 5

    filename = os.path.join(os.environ['HOME'], filename)
    epoch_size = len(h5py.File(filename))

    return filename, epoch_size, data_size


def dataset_indices(dataset_name='dev', idx=0, data_size=100, data_indices=0, index_file=''):
    if dataset_name[0:3] == 'dev':
        if data_indices == 0:  # for compatibility: continuously-indexed, fixed-sized set
            return [i for i in range(data_size*idx, data_size*(idx+1))]
        elif len(data_indices) > 1:  # specified data_indices
            return [data_indices[i] for i in range(data_size * idx, data_size * (idx + 1))]
            # return np.random.choice(epoch_size, size=dev_size, replace=False).tolist()

    elif dataset_name[0:3] == 'tot':  # only idx is useful
        file_id = h5py.File(index_file, 'r')
        return list(file_id['/data_set%03d' % idx].value.squeeze())

