import matplotlib.pyplot as plt
import torch
import numpy as np
from collections import Counter

def determine_wall_mask(h, w, deepest_layer, pred_seg, plane_mode, face, face_target, seg_maps):
    '''
    deepest_layer: 有效深度图存在的最大层数
    pred_seg: (h*w), 预测的语义分割图
    plane_mode: layout_seg中某个墙面的mask在pred_seg对应区域中mask的比值阈值，大于这个阈值才认为layout_seg中对应的mask正确
    face： 产生seg_maps的墙面的标号
    face_target: 目标墙面
    seg_maps: (h*w, num_face), 由face对应的墙面深度图相交，并排序的多层语义图

    一般情况下，face == face_target； 困难情况时，face > face_target
    '''

    num_face_target = len(face_target)

    # 先将第一层语义分割图当做初始分割图
    layout_seg = seg_maps[:, 0]

    # print(22222222)
    # print(layout_seg.shape)
    # print(pred_seg.shape)

    pred_seg = pred_seg.reshape(-1)

    for layer_id in range(1, deepest_layer):
        wrong_mask = np.zeros(h * w)    # record the mask of wrong wall

        flag = 0    # 用于判断掩模正确的墙面数量
        # check whether the layout is right
        for single_face in face_target:
            mask = layout_seg == single_face
            num_pixel_face = np.count_nonzero(mask)

            region_pred = pred_seg[mask]

            pred_counter = Counter(region_pred)
            mode_keys = list(pred_counter.keys())
            if single_face not in mode_keys:
                wrong_mask += mask
                continue

            mode_values = list(pred_counter.values())

            mode_dict = dict(zip(mode_keys, mode_values))


            if mode_dict[single_face]/num_pixel_face > plane_mode:
                flag += 1
            else:
                wrong_mask += mask

        # if all the face is right, then break
        if flag == num_face_target:
            break

        # plt.imshow(wrong_mask.reshape(h, w))
        # plt.show()
        # replace the wrong_mask in layout_seg with the corresponding value in next seg_map
        layout_seg_layer = seg_maps[:, layer_id]
        wrong_mask = wrong_mask>0
        layout_seg[wrong_mask] = layout_seg_layer[wrong_mask]


    layout_seg_ = np.zeros(h*w)
    for i in face_target:
        mask = layout_seg == i
        layout_seg_[mask] = layout_seg[mask]

    return layout_seg_




