import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
from datetime import datetime
import multiprocessing

from tensorpack import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.symbolic_functions import *

try:
    from .reader import Data
    from .pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module
    from .cfgs.config import cfg
except Exception:
    from reader import Data
    from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module
    from cfgs.config import cfg

def get_instance_seg_net(point_cloud):
    ''' 3D instance segmentation PointNet v2 network.
    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
    Output:
        logits: TF tensor in shape (B,N,2), scores for bkg/clutter and object
    '''

    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    if cfg.num_channel > 3:
        l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,cfg.num_channel-3])
    else:
        l0_points = None

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg( \
        xyz=l0_xyz, 
        points=l0_points,
        npoint=128,
        radius_list=[2,4,8],
        nsample_list=[32,64,128],
        mlp_list=[[32,32,64], [64,64,128], [64,96,128]],
        scope='layer1')
    l2_xyz, l2_points = pointnet_sa_module_msg( \
        xyz=l1_xyz,
        points=l1_points,
        npoint=32,
        radius_list=[4,8,16],
        nsample_list=[64,64,128],
        mlp_list=[[64,64,128], [128,128,256], [128,128,256]],
        scope='layer2')
    l3_xyz, l3_points, _ = pointnet_sa_module( \
        xyz=l2_xyz,
        points=l2_points,
        npoint=None,
        radius=None,
        nsample=None,
        mlp=[128,256,1024],
        mlp2=None,
        group_all=True,
        scope='layer3')

    # Feature Propagation layers
    l2_points = pointnet_fp_module( \
        xyz1=l2_xyz,
        xyz2=l3_xyz,
        points1=l2_points,
        points2=l3_points,
        mlp=[128,128],
        scope='fa_layer1')
    l1_points = pointnet_fp_module( \
        xyz1=l1_xyz,
        xyz2=l2_xyz,
        points1=l1_points,
        points2=l2_points,
        mlp=[128,128],
        scope='fa_layer2')
    l0_points = pointnet_fp_module( \
        xyz1=l0_xyz,
        xyz2=l1_xyz,
        points1=tf.concat([l0_xyz,l0_points],axis=-1),
        points2=l1_points,
        mlp=[128,128],
        scope='fa_layer3')

    # FC layers
    l0_points = tf.expand_dims(l0_points, 2)
    net = Conv2D("conv-fc1",
                 l0_points,
                 128,
                 kernel_size=1,
                 padding='VALID',
                 activation=BNReLU)
    # net = Dropout('dp1', net, rate=0.7)
    logits = Conv2D("conv-fc2",
                    net,
                    2,
                    kernel_size=1,
                    padding='VALID')
    logits = tf.squeeze(logits, axis=[2])

    return logits


def get_loss(label, logits):                                
    ''' Loss functions for 3D object detection.                       
    Input:
        label: TF int32 tensor in shape (B,N)                    
        logits: TF int32 tensor in shape (B,N)                   
    Output:
        loss: TF scalar tensor                                        
            the total_loss is also added to the losses collection     
    '''
    # 3D Segmentation loss                                            
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
        logits=logits, labels=label))
    loss = tf.identity(loss, name='loss')
    return loss


class Model(ModelDesc):
    def __init__(self):
        super(Model, self).__init__()

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, None, cfg.num_channel], 'input'),
                InputDesc(tf.int32, [None, None], 'label')]


    def _build_graph(self, inputs):
        pc, label = inputs

        logits = get_instance_seg_net(pc)

        loss = get_loss(label, logits)

        if cfg.weight_decay > 0:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
        else:
            wd_cost = tf.constant(0.0)

        self.cost = tf.add_n([loss, wd_cost], name='cost')

        pred = tf.argmax(logits, 2)

        pred = tf.identity(pred, name='pred')

        correct = tf.equal(pred, tf.to_int64(label))
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(cfg.batch_size * cfg.num_point)
        accuracy = tf.identity(accuracy, name='accuracy')

        add_moving_summary(loss, accuracy)


    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 1e-3, summary=True)
        # return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        return tf.train.AdamOptimizer(lr)


def get_data(name, batch_size):
    isTrain = name == 'train'

    if isTrain:
        ds = Data(cfg.train_ds_path, shuffle=True, random_flip=True, random_shift=True)
    else:
        ds = Data(cfg.val_ds_path, shuffle=False, random_flip=False, random_shift=False)

    if isTrain:
        ds = PrefetchDataZMQ(ds, min(8, multiprocessing.cpu_count()))
    ds = BatchData(ds, batch_size, remainder=not isTrain)
    return ds


def get_config(model, args):
    nr_tower = get_nr_gpu()

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, cfg.batch_size))
    dataset_train = get_data('train', cfg.batch_size)
    dataset_val = get_data('val', cfg.batch_size)
    callbacks = [
        ModelSaver(),
        InferenceRunner(dataset_val,
                        ScalarStats(['accuracy', 'loss'])),
        ScheduledHyperParamSetter('learning_rate',
                                 [(0, 1e-3)]),
                                 # [(0, 1e-3), (50, 5e-4), (100, 3e-4), (200, 1e-4)]),
        HumanHyperParamSetter('learning_rate'),
    ]

    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        max_epoch=300,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1', help='GPU to use [default: GPU 1]')
    parser.add_argument('--logdir', default=None, help='directory of logging')
    parser.add_argument('--load', help='load model', default=None)

    args = parser.parse_args()


    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model()

    if args.logdir != None:
        logger.set_logger_dir(os.path.join("train_log", args.logdir))
    else:
        logger.auto_set_dir()
    config = get_config(model, args)
    config.nr_tower = get_nr_gpu()

    if args.load:
        config.session_init = get_model_loader(args.load)

    trainer = SyncMultiGPUTrainerParameterServer(get_nr_gpu())
    launch_train_with_config(config, trainer)
