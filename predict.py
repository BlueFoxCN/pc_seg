import cv2
import argparse
import shutil
import pickle
import sys
import os
import numpy as np
from plyfile import PlyData

from tensorpack import *

from cfgs.config import cfg
from utils import save_ply_file, rotate_pc
from train import Model

def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input"],
                                   output_names=["pred"])

    predict_func = OfflinePredictor(predict_config)
    return predict_func

def predict_txt(txt_path, pred_func, output_dir):
    f = open(txt_path, 'r')
    lines = f.readlines()
    id_list = []
    box2d_list = []
    for line in lines:
        eles = line.strip().split(' ')
        ply_id = eles[0]
        # id_list.append(eles[0])
        box = [int(e) for e in eles[1:]]
        # self.box2d_list.append(box)

        ply_path = os.path.join(cfg.ds_dir, cfg.frustum_dir, "%s.ply" % ply_id)
        ply_data = PlyData.read(ply_path)
        vert = ply_data['vertex']
        data = [vert[e] for e in ['x', 'y', 'z', 'red', 'green', 'blue']]
        pc = np.array(data).transpose((1,0))
        pc = rotate_pc(pc, box)

        pc = np.expand_dims(pc, 0)

        predictions = pred_func(pc)[0]

        seg_idxs = np.where(predictions[0])[0]
        seg_pc = pc[0, seg_idxs]

        save_ply_file(pc[0], os.path.join(output_dir, '%s.ply' % ply_id))
        save_ply_file(seg_pc, os.path.join(output_dir, '%s_output.ply' % ply_id))



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.')
    # parser.add_argument('--input_path', help='path of the input ply file')
    parser.add_argument('--input_txt_path', help='path of the input text file')
    # parser.add_argument('--input_dir', help='path of the input dir')

    # parser.add_argument('--output_path', help='path of the output ply file', default='output.ply')
    parser.add_argument('--output_dir', help='directory to save image result', default='output')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    pred_func = get_pred_func(args)

    if args.output_dir is not None and os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    predict_txt(args.input_txt_path, pred_func, args.output_dir)


