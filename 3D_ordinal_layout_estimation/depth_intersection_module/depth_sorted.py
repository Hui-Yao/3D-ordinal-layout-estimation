import torch
import numpy as np
import matplotlib.pyplot as plt


def depth_sorted(h, w, K_inv, face_instance_nd3):

    num_plane = len(face_instance_nd3)
    depth_maps = np.zeros((num_plane, h, w))
    seg_maps = np.zeros((num_plane, h, w))

    coordi_x = np.arange(w).reshape(1, w)
    coordi_x_map = np.repeat(coordi_x, h, axis=0)  # (h, w)
    coordi_y = np.arange(h).reshape(h, 1)
    coordi_y_map = np.repeat(coordi_y, w, axis=1)  # (h, w)
    coordi_homogeneous_map = np.ones((h, w))
    coordi_map = np.concatenate(
        (coordi_x_map[np.newaxis, :, :], coordi_y_map[np.newaxis, :, :], coordi_homogeneous_map[np.newaxis, :, :]),
        axis=0)  # (h, w, 3)
    coordi_map = coordi_map.reshape(-1, h * w)  # (3, hw) of (u, v, 1)

    k_inv_uv = np.matmul(K_inv, coordi_map)  # (3, hw) of (x/z, y/z, 1)

    for face_id in range(num_plane):
        single_face = face_instance_nd3[face_id]
        nd = single_face[1:]
        _depth = np.matmul(nd, k_inv_uv)  # (a/d, b/d, c/d)*(x/z, y/z, 1) = 1/z
        single_depth_map = 1. / _depth  # z = 1/(1/z)
        mask_fu = single_depth_map < 0  # abandon the region infront of the camera plane(XY plane)
        single_depth_map[mask_fu] = 256
        single_depth_map = single_depth_map.reshape(h, w)

        depth_maps[face_id][:, :] = single_depth_map.reshape(h, w)
        seg_maps[face_id][:, :] = single_face[0]

    depth_maps = depth_maps.reshape(-1, h * w).T  # (hw, num_plane)
    seg_maps = seg_maps.reshape(-1, h * w).T  # (hw, num_plane)

    # along the channel direction, sort the depth value in ascending order, and return the sorted order
    id_maps = np.argsort(depth_maps, axis=1)  # (hw, num_plane)

    reasonable_seg_maps = np.zeros((h*w, num_plane))
    reasonable_depth_maps = np.zeros((h*w, num_plane))
    for i in range(num_plane):
        id_map = id_maps[:, i]
        seg_map = seg_maps[range(h * w), id_map]
        depth_map = depth_maps[range(h * w), id_map]
        reasonable_seg_maps[:, i] = seg_map
        reasonable_depth_maps[:, i] = depth_map

    return depth_maps, reasonable_seg_maps, id_maps