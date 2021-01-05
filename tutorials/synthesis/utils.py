import random

import h5py


class H5FrameIterator():
    def __init__(self, filename, batch_size):
        self.h5_file = h5py.File(filename,'r')
        self.num_frames = len(self.h5_file)
        self.batch_size = batch_size
        self.num_batches = int(self.num_frames/self.batch_size) # skip the 
        self.batch_idx = 0
        self.frame_indices = [i for i in range(self.num_frames)]
        random.shuffle(self.frame_indices)
    
    def __iter__(self):
        self.batch_idx += 1
        return self
    
    def __next__(self):
        batch_frame_idx = self.frame_indices[self.batch_idx*self.batch_size:(self.batch_idx+1)*self.batch_size]
        dataset = '/frame%06d' % batch_frame_idx
        if self.batch_idx==self.num_batches:
            return StopIteration
        return self.h5_file[dataset][()]
