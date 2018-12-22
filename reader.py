import cv2
import pickle
import sys
import os
import numpy as np

from tensorpack import *

from cfgs.config import cfg
from utils import save_ply_file, rotate_pc

class Data(RNGDataFlow):
    ''' Dataset class for Frustum PointNets training/evaluation.
    '''
    def __init__(self, ds_path, shuffle=False, random_flip=False, random_shift=False):
        '''
        Input:
            split: string, 'train' or 'val'
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
        '''
        self.shuffle = shuffle
        self.random_flip = random_flip
        self.random_shift = random_shift

        # self.input_list
        # self.box2d_list
        # self.label_list
        with open(ds_path, 'rb') as fp:
            self.id_list = pickle.load(fp)
            self.input_list = pickle.load(fp)
            self.box2d_list = pickle.load(fp)
            self.label_list = pickle.load(fp)

    def size(self):
        return len(self.box2d_list)

    def get_data(self):
        idxs = np.arange(len(self.box2d_list))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            # Get point cloud and label
            point_set = self.get_center_view_point_set(k)
            seg = self.label_list[k] 
            label = np.array([False] * point_set.shape[0])

            # Resample point cloud
            choice = np.random.choice(point_set.shape[0], cfg.num_point, replace=False)
            point_set = point_set[choice, :]

            # ------------------------------ LABELS ----------------------------
            label[np.array(seg)] = True
            label = label[choice].astype(np.int)

            # Data Augmentation
            if self.random_flip:
                if np.random.random() > 0.5: # 50% chance horizontal flipping
                    point_set[:,0] *= -1
            if self.random_shift:
                min_dist = np.min(point_set[:,2])
                shift = np.clip(np.random.randn() * min_dist * 0.05,
                                -min_dist * 0.2,
                                min_dist * 0.2)
                point_set[:,2] += shift

            yield [point_set, label]

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        box2d = np.copy(self.box2d_list[index])
        return rotate_pc(point_set, box2d)

if __name__=='__main__':
    ds = Data(cfg.train_ds_path, shuffle=False)
    ds.reset_state()

    g = ds.get_data()

    for i in range(10):
        dp = next(g)
        import pdb
        pdb.set_trace()
        '''
        for i in range(cfg.batch_size):
            save_ply_file(batch_data[i], 'sample_%d.ply' % i)
            save_ply_file(batch_data[i,np.where(batch_label[i])[0]], 'sample_%d_seg.ply' % i)
        '''

