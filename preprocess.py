import pickle
import shutil
from tqdm import tqdm
from scipy import misc
import cv2
import os
import numpy as np
from plyfile import PlyData, PlyElement

try:
    from .cfgs.config import cfg
    from .utils import enlarge_box, save_ply_file, filter_pc
except Exception:
    from cfgs.config import cfg
    from utils import enlarge_box, save_ply_file, filter_pc

def get_frustum_path(file_idx, box_idx, fmt='ply'):
    return "%s/%s/%s_%d.%s" % (cfg.ds_dir, cfg.frustum_dir, file_idx, box_idx, fmt)

def preprocess():
    '''
    Based on the 2d box, extract the frustum point set
    '''
    frustum_dir_path = os.path.join(cfg.ds_dir, cfg.frustum_dir)
    if os.path.isdir(frustum_dir_path):
        shutil.rmtree(frustum_dir_path)
    os.mkdir(frustum_dir_path)

    data_dir = "%s/images" % cfg.ds_dir
    label_file = "%s/detection_label.txt" % cfg.ds_dir
    
    f = open(label_file, 'r')
    labels = f.readlines()
    
    for line in tqdm(labels, ascii=True):
        eles = line.split(' ')
        file_idx = eles[0].split('/')[-1].split('.')[0]
        ply_path = os.path.join(data_dir, "%s.ply" % file_idx)
        ply_data = PlyData.read(ply_path)
        img_path = os.path.join(data_dir, "%s.jpg" % file_idx)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
    
        vert = ply_data['vertex']
        data = [vert[e] for e in ['x', 'y', 'z', 'red', 'green', 'blue']]
        pc = np.array(data).transpose((1,0))
    
        ele_idx = 1
        box_idx = 0
        while ele_idx < len(eles):
            box = [int(e) for e in eles[ele_idx:ele_idx + 4]]

            box = enlarge_box(box, cfg.xpd_ratio, h, w)

            keep = filter_pc(pc, box)
            keep_pc = pc[keep]
            save_ply_file(keep_pc, get_frustum_path(file_idx, box_idx))
    
            # draw the detection result on the 2d image and save
            img_copy = np.copy(img)
            cv2.rectangle(img_copy,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0, 0, 255),
                          3)
            cv2.imwrite(get_frustum_path(file_idx, box_idx, fmt='jpg'), img_copy)
    
            ele_idx += 5
            box_idx += 1

def extract_pc_seg():

    label_file = "%s/detection_label.txt" % cfg.ds_dir

    f = open(label_file, 'r')
    labels = f.readlines()

    label_dict = { }

    for line in labels:
        eles = line.split(' ')
        file_idx = eles[0].split('/')[-1].split('.')[0]

        ele_idx = 1
        box_idx = 0
        boxes = []
        while ele_idx < len(eles):
            box = [int(e) for e in eles[ele_idx:ele_idx + 4]]
            boxes.append(box)

            ele_idx += 5
            box_idx += 1
        label_dict[file_idx] = boxes

    input_list = []
    box2d_list = []
    label_list = []
    id_list = []

    lines = []

    segment_path = os.path.join(cfg.ds_dir, cfg.segment_dir)
    ply_files = os.listdir(segment_path)
    for ply_file in tqdm(ply_files, ascii=True):
        seg_ply_path = os.path.join(segment_path, ply_file)

        ori_ply_name = ply_file.split(' ')[0]
        ori_ply_path = os.path.join(cfg.ds_dir, cfg.frustum_dir, ori_ply_name) + ".ply"

        ori_ply_data = PlyData.read(ori_ply_path)
        seg_ply_data = PlyData.read(seg_ply_path)

        ori_vert = ori_ply_data['vertex']
        ori_data = [ori_vert[e] for e in ['x', 'y', 'z', 'red', 'green', 'blue']]
        ori_pc = np.array(ori_data).transpose((1,0))

        # confirm that the point number achieves the minimum
        if ori_pc.shape[0] < cfg.num_point:
            continue

        seg_vert = seg_ply_data['vertex']
        seg_data = [seg_vert[e] for e in ['x', 'y', 'z', 'red', 'green', 'blue']]
        seg_pc = np.array(seg_data).transpose((1,0))

        ori_p_list = ori_pc.tolist()
        seg_p_list = seg_pc.tolist()

        label = []
        for idx, ori_p in enumerate(ori_p_list):
            if ori_p in seg_p_list:
                label.append(idx)

        # label_list.append(label)
        # input_list.append(ori_pc)
        # id_list.append(ori_ply_name)

        # box2d_list.append(label_dict[file_idx][box_idx])

        f = open(os.path.join(cfg.ds_dir, cfg.frustum_dir, "%s.pkl" % ori_ply_name), 'wb')
        pickle.dump(label, f)

        file_idx, box_idx = ori_ply_name.split('_')
        box_idx = int(box_idx)
        box_ele = [str(e) for e in label_dict[file_idx][box_idx]]

        lines.append("%s %s" % (ori_ply_name, " ".join(box_ele)))

    f = open('train.txt', 'w')
    f.write('\n'.join(lines))
    
    '''
    # dump the list into pickle file
    with open(cfg.train_ds_path, 'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(input_list, fp)
        pickle.dump(box2d_list, fp)
        pickle.dump(label_list, fp)
    '''

if __name__ == "__main__":
    # preprocess()
    extract_pc_seg()

