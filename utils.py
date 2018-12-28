from plyfile import PlyData, PlyElement
import cv2
import numpy as np

try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg

def enlarge_box(box, ratio, img_h, img_w):
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    width = box[2] - box[0]
    height = box[3] - box[1]

    xmin = int(max(0, x_center - ratio * (width / 2)))
    ymin = int(max(0, y_center - ratio * (height / 2)))
    xmax = int(min(img_w - 1, x_center + ratio * (width / 2)))
    ymax = int(min(img_h - 1, y_center + ratio * (height / 2)))
    box = [xmin, ymin, xmax, ymax]

    return box


def save_ply_file(pc, file_path):
    '''
    Save point cloud data to a ply file.
    Input:
        pc: The numpy array of point cloud data. Dimension is Nx6. The second dimension is x, y, z, red, green and blue
        file_path: The ply file path
    '''
    n = pc.shape[0]

    vertex = pc[:, :3]
    vertex = np.array([tuple(e) for e in vertex], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    if cfg.num_channel == 6:
        vertex_color = pc[:, 3:]
        vertex_color = np.array([tuple(e) for e in vertex_color], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    if cfg.num_channel == 6:
        vertex_all = np.empty(n, vertex.dtype.descr + vertex_color.dtype.descr)
    else:
        vertex_all = np.empty(n, vertex.dtype.descr)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if cfg.num_channel == 6:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)

    ply.write(file_path)

def filter_pc(pc, boxes):
    '''
    filter the points by the multiple 2d boxes
    Input:
        pc: numpy array (N,C)
            z is facing forward, x is left ward, y is downward
        boxes: the boxes given by the detection model, each element is [xmin, ymin, xmax, ymax], or a single box
    Output:
        keeps: list of index list, which indicates the remained points, or a single list when a single box is provided
    '''
    single_box = False
    if isinstance(boxes, list) == False:
        single_box = True
        boxes = [boxes]

    keeps = [[] for _ in boxes]
    cvt_mat = cfg.color_mat.dot(cfg.d2c_mat)

    points_homo = np.hstack([pc[:, :3], np.ones((pc.shape[0], 1))])
    points_homo = np.transpose(points_homo)

    color_uvs_homo = cvt_mat.dot(points_homo)
    color_uvs = color_uvs_homo[:2] / color_uvs_homo[2]

    us = color_uvs[0]
    vs = color_uvs[1]

    for box_idx, box in enumerate(boxes):
        indicators = np.all([box[0] <= us, us <= box[2], box[1] <= vs, vs <= box[3]], axis=0)
        keeps[box_idx] = np.where(indicators)[0].tolist()

    return keeps[0] if single_box else keeps


def rotate_pc(pc, box2d):
    '''
    rotate the frame to make z-axis through the center of the 2d box
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        box2d: the box given by the detection model, [xmin, ymin, xmax, ymax]
    Output:
        pc: updated pc with XYZ rotated
    '''
    xcenter = (box2d[0] + box2d[2]) / 2
    ycenter = (box2d[1] + box2d[3]) / 2
    dist = np.sqrt((xcenter - cfg.intr['cx']) ** 2 + (ycenter - cfg.intr['cy']) ** 2)
    f_mean = (cfg.intr['fx'] + cfg.intr['fy']) / 2
    angle = -np.arctan2(dist, f_mean)
    vec = np.array([xcenter - cfg.intr['cx'], ycenter - cfg.intr['cy'], 0])
    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, vec)
    axis = axis / np.linalg.norm(axis) * angle
    rot_mat = cv2.Rodrigues(axis)[0]

    for point in pc:
        coord = point[:3]
        new_coord = np.matmul(rot_mat, coord)
        point[:3] = new_coord

    return pc
