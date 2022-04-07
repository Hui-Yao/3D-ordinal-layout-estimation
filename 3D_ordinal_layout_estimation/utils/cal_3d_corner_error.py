import torch
import numpy as np

def corner_error(h, w, k_inv, layout_corner_xyz, gt_keypoint):
    '''
    func: cal the 3d corner error

    layout_corner_xyz: the pred (x, y, z)coordinate of keypoint
    gt_keypoint:  the GT (u, v, z) coordinate of keypoint
    '''

    num_point_ = 0
    for i in layout_corner_xyz:
        if i[2] != 0:
            num_point_ += 1
    pred_keypoint_xyz = layout_corner_xyz[0:num_point_, :]

    num_point = 0
    for i in gt_keypoint:
        if i[2] != 0:
            num_point += 1

    gt_keypoint_uvz = gt_keypoint[0:num_point, :]
    gt_keypoint_uvz[:, 0] = gt_keypoint_uvz[:, 0]*w/640
    gt_keypoint_uvz[:, 1] = gt_keypoint_uvz[:, 1]*h/480
    # print(gt_keypoint)

    # gt_z = np.zeros(num_point)
    gt_z = gt_keypoint_uvz[:, 2].copy()
    gt_keypoint_uvz[:, 2] = 1
    gt_uv1 = gt_keypoint_uvz.T

    gt_keypoint_xyz = np.matmul(k_inv, gt_uv1) * gt_z[:, np.newaxis].T
    gt_keypoint_xyz = gt_keypoint_xyz.T


    # print('gt_keypoint:\n', gt_keypoint_xyz)
    # print(gt_keypoint_xyz.shape)
    # print('pred_keypoint\n',pred_keypoint_xyz)
    # print(pred_keypoint_xyz.shape)

    gt_keypoint_xyz = torch.tensor(gt_keypoint_xyz)
    pred_keypoint_xyz = torch.tensor(pred_keypoint_xyz)

    m = gt_keypoint_xyz.size(0)
    n = pred_keypoint_xyz.size(0)

    gt_repeat = gt_keypoint_xyz.repeat(1, n).view(n * m, 3)
    pred_repeat = pred_keypoint_xyz.repeat(m, 1)

    distance = torch.nn.PairwiseDistance(keepdim=True, p=2)(gt_repeat, pred_repeat).view(m, n)

    error = torch.min(distance, dim=1)[0]
    error_mean = torch.mean(error)  # the mean 3d corner error for single keypoint

    return error_mean
