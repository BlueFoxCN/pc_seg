import cv2
import pickle
import sys
import os
import numpy as np
from plyfile import PlyData

from tensorpack import *

try:
    from .cfgs.config import cfg
    from .utils import save_ply_file, rotate_pc
except Exception:
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

        f = open(ds_path, 'r')
        lines = f.readlines()

        self.id_list = []
        self.box2d_list = []
        for line in lines:
            eles = line.split(' ')
            self.id_list.append(eles[0])
            box = [int(e) for e in eles[1:]]
            self.box2d_list.append(box)

        # self.input_list
        # self.box2d_list
        # self.label_list
        '''
        with open(ds_path, 'rb') as fp:
            self.id_list = pickle.load(fp)
            self.input_list = pickle.load(fp)
            self.box2d_list = pickle.load(fp)
            self.label_list = pickle.load(fp)
        '''

    def size(self):
        return len(self.box2d_list)

    def get_data(self):
        idxs = np.arange(len(self.box2d_list))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            # Get point cloud and label
            point_set = self.get_center_view_point_set(k)

            f = open(os.path.join(cfg.ds_dir, cfg.frustum_dir, "%s.pkl" % self.id_list[k]), 'rb')
            seg = pickle.load(f)
            # seg = self.label_list[k] 
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
        ply_path = os.path.join(cfg.ds_dir, cfg.frustum_dir, "%s.ply" % self.id_list[index])
        ply_data = PlyData.read(ply_path)
        vert = ply_data['vertex']
        data = [vert[e] for e in ['x', 'y', 'z', 'red', 'green', 'blue']]
        pc = np.array(data).transpose((1,0))

        # Use np.copy to avoid corrupting original data
        # point_set = np.copy(self.input_list[index])
        # box2d = np.copy(self.box2d_list[index])
        return rotate_pc(pc, self.box2d_list[index])

if __name__=='__main__':
    ds = Data('train.txt', shuffle=False)
    ds.reset_state()

    g = ds.get_data()

    for i in range(10):
        dp = next(g)
        save_ply_file(dp[0], 'sample_%d.ply' % i)
        save_ply_file(dp[0][np.where(dp[1])[0]], 'sample_%d_seg.ply' % i)

