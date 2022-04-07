import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# own
from depth_intersection_module.cal_frustum_planes import calc_frustum_planes
from depth_intersection_module.misc import united
from depth_intersection_module.depth_sorted import depth_sorted
from depth_intersection_module.determine_wall_mask import determine_wall_mask
from depth_intersection_module.intersect_planes import intersect_planes

from pprint import pprint

class DepthIntersection(nn.Module):
    def __init__(self, cfg):
        super(DepthIntersection, self).__init__()
        self.h, self.w = cfg.dataset.h, cfg.dataset.w
        self.minimum_plane_ratio = 0.01   # the minimum ratio of pixel to be a plane
        self.plane_mode = 0.4
        self.dilate_size = 5
        self.depth_threshold_ratio = 0.03

    def forward(self, image, intrinsic, pred_prob, pred_seg, pred_surface_normal):
        '''
        image: the image using for predict
        pred_prob: (num_class, h, w), the prob of a pixel belong to a certein class
        pred_seg: (h, w), the pred semantic map
        pred_surface_normal: (3, h, w)
        '''

        h, w = self.h, self.w
        num_pixel_picture = h*w
        plane_mode = self.plane_mode
        dilate_size = self.dilate_size
        depth_threshold_ratio = self.depth_threshold_ratio
        K_inv = np.linalg.inv(intrinsic)

        pred_prob = pred_prob.reshape(-1, h*w)
        pred_prob = F.normalize(pred_prob, p=1, dim=1)
        pred_prob = pred_prob.cpu().numpy()
        pred_seg = pred_seg.cpu().numpy().reshape(-1)
        pred_surface_normal = pred_surface_normal.cpu().numpy().reshape(3, h*w)

        # get the frustum plane
        frustum_planes = calc_frustum_planes(h, w, intrinsic)  # clockwise. up, right, down, left

        # 以1%为阈值，求出当前图片中存在的平面
        face = []
        filted_prob = []
        for i in range(7):
            mask = pred_seg == i
            num_pixel_class = np.count_nonzero(mask)
            if num_pixel_class/num_pixel_picture > self.minimum_plane_ratio:
                face.append(i)
                filted_prob.append(pred_prob[i][np.newaxis, :])
        face = np.array(face)
        face_wall = [i for i in face if i >1]
        face_cf  = [i for i in face if i in [0, 1]]
        num_face = len(face)
        num_wall = len(face_wall)

        filted_prob = np.concatenate(filted_prob, axis=0)

        # using the pred prob and pred plane param, through linear algebra to calculate the instance plane param
        pred_surface_normal = pred_surface_normal.T

        instance_nd3 = np.matmul(filted_prob, pred_surface_normal)     # (k, 3), k is 7, 3 represent (a/d, b/d, c/d)

        face_instance_nd3 = np.concatenate([face[:, np.newaxis], instance_nd3], axis=1)

        # turns plane params for (a/d, b/d, c/d) to (a, b, c, d)
        _plane_d = np.sqrt(np.sum(np.square(instance_nd3), axis=1))[:, np.newaxis]  #
        plane_d = 1./_plane_d
        abc = instance_nd3/_plane_d
        face_instance_nd4 = np.concatenate([face[:, np.newaxis], abc, plane_d], axis=1)    # (class, a, b, c, d)


        # get the label of wall with max offset
        wall_instance_nd4 = [i for i in face_instance_nd4 if i[0] > 1]
        for itr, wall_nd in enumerate(wall_instance_nd4):
            if itr == 0:
                # wall_max_offset_label = wall_nd[0]
                max_offset = wall_nd[4]
                wall_max_offset_id = itr
            if wall_nd[4] > max_offset:
                # wall_max_offset_label = wall_nd[0]  # svae the semantic label of max offset
                max_offset = wall_nd[4]  # save the max offset
                wall_max_offset_id = itr  # svae the id of max offset

        deepest_layer = int(np.ceil(num_wall/2))

        if num_wall < 5 or wall_max_offset_id<2 or wall_max_offset_id+2>4:
            # if num_wall less then 5, directly using depthmap intersection
            # if the wall with max offset is the first and last two wall, then can directly use depth intersection
            print('Plane A')

            # id_maps is the ascending order of the depth value of a certein pxiel,
            # seg_maps is the semantic label after sorted, depth_maps is not sorted
            depth_maps, seg_maps, id_maps = depth_sorted(h, w, K_inv, face_instance_nd3)

            layout_seg = determine_wall_mask(h, w, deepest_layer, pred_seg, plane_mode, face, face, seg_maps)

        else:
            # if num_wall is more than 5, seperate the wall to two part from the wall with max offset, and doing depth intersection twice
            print('Plane B')

            # get the ceil and floor
            instance_nd3_cf = [i for i in face_instance_nd3 if i[0] in [0, 1]]
            face_cf = [i for i in face if i in [0, 1]]
            num_cf = len(face_cf)

            wall_instance_nd3 = [i for i in face_instance_nd3 if i[0] > 1]

            # get the plane paramters for part1 and part2
            instance_nd_part1 = wall_instance_nd3[:wall_max_offset_id + 2] + instance_nd3_cf
            instance_nd_part2 = wall_instance_nd3[wall_max_offset_id :] + instance_nd3_cf
            face_wall_part1 = face_wall[:wall_max_offset_id + 2] + face_cf
            face_wall_part2 = face_wall[wall_max_offset_id : ] + face_cf

            face_target_part1 = face_wall[:wall_max_offset_id + 1]
            face_target_part2 = face_wall[wall_max_offset_id + 1:]

            # get the mask of wall from first wall to the wall with max offset(exclude ceil and floor)
            depth_map2, seg_map2, id_map2 = depth_sorted(h, w, K_inv, instance_nd_part1)
            layout_seg1 = determine_wall_mask(h, w, deepest_layer, pred_seg, plane_mode,
                                              face_wall_part1, face_target_part1,seg_map2)

            layout_seg3_ = np.zeros(h * w)
            depth_map3, seg_map3, id_map3 = depth_sorted(h, w, K_inv, instance_nd_part2)
            layout_seg2 = determine_wall_mask(h, w, deepest_layer, pred_seg, plane_mode,
                                              face_wall_part2, face_target_part2, seg_map3)
            # plt.subplot(152)
            # plt.imshow(layout_seg2.reshape(h, w))
            # plt.show()

            layout_seg_wall = layout_seg1 + layout_seg2
            # plt.subplot(153)
            # plt.imshow(layout_seg_wall.reshape(h, w))

            # combine the mask of walls, doing last depth intersection(include ceil and floor)
            depth_maps, seg_maps, id_maps = depth_sorted(h, w, K_inv, face_instance_nd3)

            # get the mask of ceil and floor in the reasonable depth map
            layout_seg = np.zeros(h * w)    ##########################这里有个问题，0与天花板标签相同，不知道会不会有问题######################
            for layer_id in range(deepest_layer):
                seg_map = seg_maps[:, layer_id]
                mask_ceil = seg_map == 0
                mask_floor = seg_map == 1
                layout_seg[mask_ceil] = seg_map[mask_ceil]
                layout_seg[mask_floor] = seg_map[mask_floor]
            # plt.subplot(154)
            # plt.imshow(layout_seg.reshape(h, w))
            # plt.show()

            # part of the mask of c.f. maybe wrong, but the mask of wall must be right ,
            # so replace corresponding region in layout_seg by layout_wall
            mask_wall = layout_seg_wall != 0
            layout_seg[mask_wall] = layout_seg_wall[mask_wall]

            # plt.subplot(155)
            # plt.imshow(layout_seg.reshape(h, w))
            # plt.show()

        layout_depth = np.zeros(h*w)

        for face_id, single_face in enumerate(face):
            mask = layout_seg == single_face
            layout_depth[mask] = depth_maps.T[face_id][mask]

        layout_seg = layout_seg.reshape(h, w)
        layout_depth = layout_depth.reshape(h, w)

        # get the mask of every plane, for the purpose of filt layout corner
        plane_masks = np.zeros((num_face, h, w))
        for face_id, single_face in enumerate(face):
            mask = layout_seg == single_face
            mask = np.array(mask, dtype=np.uint8)
            # every mask should be dilate
            kernel = np.ones((dilate_size, dilate_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            plane_masks[face_id] = mask

        all_planes = np.concatenate((face_instance_nd4[0:num_face], frustum_planes), axis=0)

        # get all the underlying corner
        intersection_dict_list = intersect_planes(all_planes, h, w, intrinsic)

        xyz = intersection_dict_list['xyz']  # coordinate in camera coordi system
        uvz = intersection_dict_list['uvz']  # coordinate in pixel CS, ~
        round_uvz = intersection_dict_list['round_uvz']  # round_uvz = [round(u), round(v), z]
        origin_plane_id = intersection_dict_list['plane_id']  # indicate every corner is generated by which three plane

        # layout_corner_round = np.zeros((20, 3))
        layout_corner_float = np.zeros((20, 3))
        layout_corner_xyz = np.zeros((20, 3))
        num_corner = 0
        # filt the origin layout corner by two role
        for point_id, point in enumerate(round_uvz):
            u_, v_, z_ = point

            u_ = int(u_)  # only for windows
            v_ = int(v_)
            # first role: the correct point should lie on the depth map
            depth_distance_threshold = z_ * depth_threshold_ratio

            if abs(layout_depth[v_, u_] - z_) < depth_distance_threshold:

                plane_ids = origin_plane_id[point_id]  # plane_ids include frustum plane
                valid_plane_ids = [i for i in plane_ids if i < num_face]  # plane_ids exclude frustum plane
                valid_flags = np.zeros(len(valid_plane_ids))  # 用来判断corner是不是在平面上的标志

                # second role: corner应该在相交得到这三个点的平面掩模上
                for id, point_plane_id in enumerate(valid_plane_ids):
                    mask = plane_masks[point_plane_id]
                    if mask[v_, u_] != 0:  # corner是否在某个对应的plane_mask上
                        valid_flags[id] = 1
                    if np.sum(valid_flags) == len(valid_plane_ids):  # corner是否在所有对应的plane_mask上
                        point_uvz = uvz[point_id]
                        for component_id, component_coor in enumerate(point_uvz):  # 有些在视锥体上的交点的u或v值可能是一个很小很小的负数，直接令其为0
                            if component_coor < 0.001 and component_coor > -0.001:
                                point_uvz[component_id] = 0
                        # layout_corner_round[num_corner] = round_uvz[point_id]
                        layout_corner_float[num_corner] = point_uvz
                        layout_corner_xyz[num_corner] = xyz[point_id]
                        num_corner += 1

        # layout_corner_round = np.array(layout_corner_round, dtype=np.float32)
        layout_corner_float = np.array(layout_corner_float, dtype=np.float32)
        layout_corner_xyz = np.array(layout_corner_xyz, dtype=np.float32)
        # num_corner = len(layout_corner_float)

        return layout_seg, layout_depth, layout_corner_float, layout_corner_xyz


