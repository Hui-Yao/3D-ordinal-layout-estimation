import matplotlib.pyplot as plt
import torch
import copy
from skimage.filters import sobel
import open3d as o3d
import cv2
import numpy as np

def united(instance_nd3):
    # trans the plane param of (a/d, b/d, c/d) to (a, b, c, d)
    square_sum = torch.sqrt(instance_nd3[0]*instance_nd3[0] + instance_nd3[1]*instance_nd3[1] + instance_nd3[2]*instance_nd3[2])
    a = instance_nd3[0]/square_sum
    b = instance_nd3[1]/square_sum
    c = instance_nd3[2]/square_sum
    d = 1/ square_sum
    return torch.tensor([a, b, c, d])

def edge_embedding(image_, layout_seg, color):
    # detect the layout edge, and plot replace the corrsponding pixel in the image
    image = copy.copy(image_)
    color_dict = {'red': [255, 0, 0], 'green': [0, 255, 0], 'blue': [0, 0, 255], 'yellow': [255, 255, 0],
                  'cyan':[0, 255, 255], 'purple': [138, 43, 226], 'gary': [192, 192, 192]}
    color_value = color_dict[color]

    layout_edge = sobel(layout_seg)
    edge_mask = layout_edge>0

    edge_mask = edge_mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    edge_mask = cv2.dilate(edge_mask, kernel=kernel, iterations=3)
    edge_mask = edge_mask > 0

    image[:, :, 0][edge_mask] = color_value[0]
    image[:, :, 1][edge_mask] = color_value[1]
    image[:, :, 2][edge_mask] = color_value[2]

    return image

def seg_color_trans(layout_seg):
    color_dict = {0: [0, 0, 255], 1: [127, 255, 0], 2: [0, 255, 255], 3:[255, 255, 0],
                  4: [227, 23 ,13], 5: [218, 112, 214], 6: [255, 153, 87]}



    h, w = layout_seg.shape
    trans_seg = np.zeros((h, w, 3))

    for i in range(7):
        if i not in layout_seg:
            continue
        color_value = color_dict[i]
        mask = layout_seg == i

        trans_seg[:, :, 0][mask] = color_value[0]
        trans_seg[:, :, 1][mask] = color_value[1]
        trans_seg[:, :, 2][mask] = color_value[2]

    return trans_seg

def label_trans_one_side(gt_seg, face, instance_plane_params):
    '''  trans the label of 3~7 to 34576'''
    face = [i for i in face if i > 0]
    face_cf = [i - 1 for i in face if (i < 4) and (i > 0)]  # (1, 3) -->  (0,2)
    face_wall = [i - 2 for i in face if i > 3]  # (4,5,6,7,8,9) -- >(2, 3,4,5,6,7)
    num_wall = len(face_wall)

    trans_seg = np.zeros_like(gt_seg)

    for itr, single_face in enumerate(face_cf):  # ???????2??1
        if single_face == 2:
            face_cf[int(itr)] = 1

    face_wall_new = face_wall
    trans_face = np.array(face_cf + face_wall_new)

    face_filled = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10])
    for index, single_face in enumerate(face):
        wall_mask = gt_seg == single_face
        # print(111, single_face)
        # print(222, np.count_nonzero(wall_mask))
        trans_seg[wall_mask] = trans_face[index]
        face_filled[index] = trans_face[index]

    instance_plane_params[:, 0] = face_filled.reshape(9)

    return face_filled, instance_plane_params, trans_seg

def get_layout_pc(img, K, depth_from_planes, layout_seg):
    # visualize the 3d model of layout, code from RandC

    layout_edge = sobel(layout_seg)
    mask = layout_edge>0

    layout_image = img.astype(np.uint8)

    cv2.rectangle(layout_image, (0, 0), (255, 191), (0, 255, 0), 2)

    layout_image[:, :, 0][mask] = 0
    layout_image[:, :, 1][mask] = 255
    layout_image[:, :, 2][mask] = 0

    # Layout Point Cloud
    o3d_im_poly = o3d.geometry.Image((layout_image).astype(np.uint8))
    o3d_depth_from_planes_poly = o3d.geometry.Image(depth_from_planes.astype("float32"))
    camera_intr = o3d.camera.PinholeCameraIntrinsic(img.shape[1], img.shape[0], K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    o3d_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_im_poly, o3d_depth_from_planes_poly, depth_scale=1., convert_rgb_to_intensity=False,
        depth_trunc=depth_from_planes.max()+0.01)
    o3d_pcd_layout = o3d.geometry.PointCloud.create_from_rgbd_image(
        o3d_rgbd_image, camera_intr)


    return o3d_pcd_layout

def get_clutter_pc(image, intrinsic, gt_depth, layout_depth):
    # Get helpers
    img = image
    K = intrinsic
    # clutter_mask = self.clutter_mask

    clutter_thresh = 0.04
    # 布局深度图小于原始深度图的部分认为是 布局的分割掩模
    layout_mask = (layout_depth - gt_depth) < layout_depth * clutter_thresh
    clutter_mask = ~layout_mask


    # plt.imshow(layout_mask+0)
    # plt.show()

    kernel = np.ones((11, 11), np.uint8)
    clutter_mask = cv2.erode(clutter_mask.astype(np.uint8), kernel, iterations=1).astype(np.bool)
    layout_mask = cv2.erode(layout_mask.astype(np.uint8), kernel, iterations=1).astype(np.bool)

    clutter_image = img

    boundary_mask = np.zeros_like((clutter_mask))
    boundary_mask[10:-10, 10:-10] = 1
    o3d_im = o3d.geometry.Image((clutter_image).astype(np.uint8))
    depth_clutter = (boundary_mask[:, :] * clutter_mask * gt_depth).astype("float32")

    o3d_depth = o3d.geometry.Image(depth_clutter)
    camera_intr = o3d.camera.PinholeCameraIntrinsic(clutter_image.shape[1], clutter_image.shape[0], K[0, 0],
                                                    K[1, 1], K[0, 2], K[1, 2])
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_im, o3d_depth, depth_scale=1., convert_rgb_to_intensity=False,
        depth_trunc=gt_depth.max()+0.1)
    pcd_clutter = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intr)

    return pcd_clutter
