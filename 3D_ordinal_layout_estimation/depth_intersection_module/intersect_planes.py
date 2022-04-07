import numpy as np
import matplotlib.pyplot as plt

def intersect_3_planess(param_1, param_2, param_3):
    param_1 = np.array(param_1)
    param_2 = np.array(param_2)
    param_3 = np.array(param_3)

    normal_1 = param_1[1:4]
    normal_2 = param_2[1:4]
    normal_3 = param_3[1:4]
    offset_1 = param_1[4]
    offset_2 = param_2[4]
    offset_3 = param_3[4]

    normal = np.array([normal_1, normal_2, normal_3])
    offset = np.array([offset_1, offset_2, offset_3]).reshape(3, -1)

    p = np.linalg.solve(normal, offset)
    p = np.squeeze(p, 1)    # dim form (3,1) to (3)

    return  p


def intersect_3_planes(layout_planes, i, j, k, h, w, intrinsic):
    '''
    Intersect 3 planes

    :param layout_planes:
    :param i: index of plane 1
    :param j: index of plane 2
    :param k: index of plane 3
    :return:
    '''

    K = intrinsic

    j_is_frustum = layout_planes[j][0] == 0
    k_is_frustum = layout_planes[k][0] == 0

    plane1 = layout_planes[i][1: ]
    plane2 = layout_planes[j][1: ]
    plane3 = layout_planes[k][1: ]


    intersection_3d = intersect_3_planess(layout_planes[i], layout_planes[j], layout_planes[k])     # 角点的 X, Y ,Z坐标

    intersection_image_plane = np.dot(K, np.reshape(intersection_3d, (3, 1)))  # 交点的（uz, vz, z）坐标

    intersection_norm = (intersection_image_plane / (intersection_image_plane[2] + 1e-6))   # (u, v, 1)

    x = intersection_norm[0][0]
    y = intersection_norm[1][0]
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    x = np.clip(x, a_min=-1e6, a_max=1e6)
    y = np.clip(y, a_min=-1e6, a_max=1e6)

    round_x = int(np.round(x))
    round_y = int(np.round(y))
    ceil_x = int(np.ceil(x))
    ceil_y = int(np.ceil(y))

    uvz = np.array([x, y, intersection_3d[2]], dtype=np.float32)
    round_uvz = np.array([round_x, round_y, intersection_3d[2]])


    if (intersection_3d[2] > max_depth_thresh or intersection_3d[2] < min_depth_thresh):
        # print("Intersection point is far away")
        return None

    if not ((0 <= round_x <= w-1) and (0 <= round_y <= h-1)):
        return None


    intersection_dict = {}

    intersection_dict['xyz'] = intersection_3d      # coordinate of layout corner in camera coordinate system
    intersection_dict['uvz'] = uvz                  # uv stand for the pixel coordinate, z is the depth value
    intersection_dict['round_uvz'] = round_uvz      # uv after round,  z is the depth value
    intersection_dict['plane_id'] = [i, j, k]
    return intersection_dict


def intersect_planes(layout_planes, h, w, intrinsic):    # layout_planes: instance_nd5 + frustum_planes

    intersection_dict_list = {'xyz': [], 'uvz': [], 'round_uvz': [], 'plane_id': []}

    # Intersect plane triplets
    # LayoutPlane数据排列是：非视锥体平面在前，4个视锥体平面在后
    for i, layout_plane1 in enumerate(layout_planes):
        if layout_plane1[0] < 0:     # 如果第一个平面都是视锥体平面了，没必要往后做了,因为四个视锥体平面是排在最后的
            continue

        for j, layout_plane2 in enumerate(layout_planes):
            if j > len(layout_planes) - 1:  # j不能是最后一个平面， 不然k没得取值了
                continue
            if j <= i:  # j取i后面的平面
                continue
            for k, layout_plane3 in enumerate(layout_planes):  # k取i， j后面的平面
                if k <= i or k <= j:  # k取i， j后面的平面
                    continue

                # print('i, j, k:', i, j, k)
                intersection_dict = intersect_3_planes(layout_planes, i, j, k, h, w, intrinsic)
                # intersection_dict = intersect_3_planes(layout_planes, i, j, k, h, w, intrinsic)

                if intersection_dict is None:
                    continue


                intersection_dict_list['xyz'].append(intersection_dict['xyz'])
                intersection_dict_list['uvz'].append(intersection_dict['uvz'])
                intersection_dict_list['round_uvz'].append(intersection_dict['round_uvz'])
                # intersection_dict_list['plane_id'].append([layout_plane1[0], layout_plane2[0], layout_plane3[0]])
                intersection_dict_list['plane_id'].append(intersection_dict['plane_id'])



    # print("Number of intersections found: " + str(len(intersection_dict_list['round_uvz'])))
    rl_intersection_dict_list = intersection_dict_list

    intersection_dict_list['xyz'] = np.array(intersection_dict_list['xyz'], dtype=np.float32)
    intersection_dict_list['uvz'] = np.array(intersection_dict_list['uvz'], dtype=np.float32)
    intersection_dict_list['plane_id'] = np.array(intersection_dict_list['plane_id'], dtype=np.int64)

    return intersection_dict_list


max_depth_thresh = 15
min_depth_thresh = 0
par_thresh = 1e-2