from easydict import EasyDict as edict
import os
import numpy as np
import math

cfg = edict()

cfg.intr = { "fx": 578.501,
             "fy": 578.501,
             "cx": 323.388,
             "cy": 252.487 }

cfg.batch_size = 4
cfg.val_batch_size = 1
cfg.num_channel = 6
cfg.num_point = 1024
cfg.num_classes = 2

cfg.weight_decay = 0

cfg.depth_mat = np.array([[578.501, 0,       323.388, 0],
                          [0,       578.501, 252.487, 0],
                          [0,       0,       1,       0],
                          [0,       0,       0,       1]])

cfg.color_mat = np.array([[518.468, 0,       312.658, 0],
                          [0,       518.468, 239.076, 0],
                          [0,       0,       1,       0],
                          [0,       0,       0,       1]])

cfg.d2c_mat = np.array([[ 0.999996,   -0.00226997,  0.00178487, -25.179],
                        [ 0.00227352,  0.999995,   -0.00198852, -0.102628],
                        [-0.00178035,  0.00199257,  0.999996,    0.314967],
                        [0,            0,           0,           1]])

cfg.ds_dir = "dataset"
cfg.frustum_dir = "frustum_pc"
cfg.segment_dir = "frustum_pc_seg"

cfg.train_ds_path = os.path.join(cfg.ds_dir, "data.pkl")
cfg.val_ds_path = os.path.join(cfg.ds_dir, "data.pkl")

cfg.xpd_ratio = 1.1
