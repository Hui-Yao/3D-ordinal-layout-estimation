import numpy as np
# import skimage.io as io
from skimage import filters
import matplotlib.pyplot as plt
import cv2
from matplotlib import font_manager
import torch
import torchvision.transforms as tf
from PIL import Image
# from skimage.transform import rescale
# import torch
import torch.nn.functional as F
# import time
import os
import random
import skimage.io as io
import albumentations
import random
import copy


class DatasetProcess():
    def __init__(self, h, w):
        # self.h = h
        # self.w = w

        # plane fitting hyperparameter
        self.numPlanesPerSegment = 5  # 这个干啥用的？
        self.planeAreaThreshold = 10  # 如果一个组包含的点少于10个，则不认为它是一个平面
        self.numIterations = 100  # 对一个平面点组中随机抽取三个点进行平面参数拟合，并进行参数替换。最多拟合100次
        self.planeDiffThreshold = 0.01  # 平面拟合误差(aX/d+bY/d+cZ/d - 1)
        self.fittingErrorThreshold = 0.01

    def united(self, plane):
        square_sum = np.sqrt(plane[0]*plane[0] + plane[1]*plane[1] + plane[2]*plane[2])
        a = plane[0]/square_sum
        b = plane[1]/square_sum
        c = plane[2]/square_sum
        d = 1/ square_sum
        return [a, b, c, d]

    def fitPlane(self, points):   # 输入的是n个三维点的xyz坐标，shape = (n, 3)
        # 刚开始输出linalg.lstlg,然后输出slolve，再又是lstlg,再solve,最后是lstlg
        if points.shape[0] == points.shape[1]:  # 如果直接拟合不成，也就是c=2的情况，
            return np.linalg.solve(points, np.ones(points.shape[0]))
        else:   # 直接拟合所有点， matterport-3d layout全都走的是这个
            return np.linalg.lstsq(points, np.ones(points.shape[0]), rcond=None)[0]

    def cal_abcd2(self, face, layout_seg, layout_depth, intrinsics, image):
        # 相机内参
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        u0 = intrinsics[0][2]
        v0 = intrinsics[1][2]

        normald_5 = []
        normald_4 = []
        normald_dict = {}
        non_zero_index = np.nonzero(face)
        face = face[non_zero_index]
        for face_index in face:
            # 求对应区域的mask
            mask = layout_seg == face_index
            mask = mask + 0
            # mask中非零元的x，y坐标
            nonzero_tuple = np.nonzero(mask)
            X = nonzero_tuple[1]    # u
            Y = nonzero_tuple[0]    # v

            # 创建初始非零点数组
            XYZ = np.ones((len(X), 3))
            XYZ[:, 0] = X  # assign u
            XYZ[:, 1] = Y  # assign v
            for i, j in enumerate(zip(X, Y)):
                XYZ[i, 2] = layout_depth[j[1], j[0]]  # assign Z

            XYZ[:, 0] = (XYZ[:, 0]-u0) * XYZ[:, 2] / fx  # from u to X   through intrinsics
            XYZ[:, 1] = (XYZ[:, 1]-v0) * XYZ[:, 2] / fy  # from v to Y

            for c in range(2):
                if c == 0:
                    ## First try to fit one plane to see if the entire segment is one plane
                    plane = self.fitPlane(XYZ)  # 返回的是用最小二乘法拟合n个三维点，所得到的回归系数，这个plane不就是法线么
                    diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)
                    # 计算拟合出的直线与1的差距/二范数，
                    if diff.mean() < self.fittingErrorThreshold:  # 如果拟合误差小于阈值(0.05)，就吧这些点看做是同一个平面上的
                        break
                else:
                    # Run ransac
                    # print('c = 1, cant fit immediatly, runned RANSAC')
                    for planeIndex in range(self.numPlanesPerSegment):  # num~segment = 2
                        if len(XYZ) < self.planeAreaThreshold:  # 如果一个平面上点的数量少于10个，则不认为他是一个平面
                            continue
                        bestPlaneInfo = [None, 0, None]
                        for iteration in range(min(XYZ.shape[0], self.numIterations)):  # numIteration = 100，也就是做最多迭代100次
                            sampledPoints = XYZ[np.random.choice(np.arange(XYZ.shape[0]), size=(3), replace=False)]  # 从所有点中随机抽取3个点
                            try:
                                plane = self.fitPlane(sampledPoints)
                                pass
                            except:
                                continue
                            diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)  # 计算误差
                            inlierMask = diff < self.planeDiffThreshold  #   平面上符合拟合出的法线的点mask， 也就是内点的mask
                            # inlierMask.shape = (？, 3)
                            numInliers = inlierMask.sum()   # 内点的数量

                            # 如果新拟合的plane参数比上一个参数更合理（有更多的true值），则吧当前的plane参数当做最优解
                            if numInliers > bestPlaneInfo[1]:
                                bestPlaneInfo = [plane, numInliers, inlierMask]
                                pass
                            continue

                        if bestPlaneInfo[1] < self.planeAreaThreshold:  # 如果组成平面的点数少于10，则不执行后续
                            break

                        bestPlane = self.fitPlane(XYZ[bestPlaneInfo[2]])  # 用取到的inlier去拟合平面，得到更优的值
                        # plane = bestPlane[1]    # 最后用所有的内点再去拟合一次
                        plane = bestPlane    # 最后用所有的内点再去拟合一次
                    # print('num_face:', len(face))
                    # print(31, face_index)
                    # print(32, plane)
                    # abcd1 = self.united(plane)  # list, [a, b, c, d]
                    # print('united_normal:',abcd1)
                    # print('\n')
                    # plt.subplot(131)
                    # plt.imshow(image)
                    # plt.subplot(132)
                    # plt.imshow(layout_seg)
                    # plt.subplot(133)
                    # plt.imshow(layout_depth)
                    # plt.show()

            # print(31, face_index.dtype)
            # print(31, type(face_index))
            # print(31, face_index)
            # print(32, plane.dtype)
            # print(32, type(plane))
            # print(32, plane)
            # print('\n')

            face_normal_4 = [face_index] + list(plane)    # (a, b, c)

            normald_4.append(face_normal_4)

            abcd = self.united(plane)    # list, [a, b, c, d]

            face_normal_5 = [face_index] + abcd
            normald_5.append(face_normal_5)

            normald_dict[face_index] = abcd

        # transfer [face, a, b, c, d] into ndarray
        max_plane_in_dataset = 8
        tmp1 = np.zeros((max_plane_in_dataset, 4))
        tmp2 = np.zeros((max_plane_in_dataset, 5))
        for i, j in enumerate(normald_4):
            tmp1[i, :len(j)] = j
        normald_4 = tmp1
        for i, j in enumerate(normald_5):
            tmp2[i, :len(j)] = j
        normald_5 = tmp2

        return normald_4, normald_5, normald_dict

    def get_instance_masks(self, gt_seg, face):

        h, w = gt_seg.shape
        max_num_plane = np.count_nonzero(face)
        instance_masks = np.zeros((8, h, w))

        for i in range(max_num_plane):
            mask = gt_seg == face[i]
            # if np.sum(mask)>plane_pixel_threshold:
            instance_masks[i] = mask
        # num_plane += 1
        instance_masks = torch.from_numpy(instance_masks)

        return instance_masks

    def get_instance_masks2(self, gt_seg, face):
        '''
        if unknow the number of plane, using this function
        :param gt_seg:
        :param face:
        :return:
        '''

        h, w = gt_seg.shape
        plane_pixel_threshold = 0.04   # 平面像素比例的最小阈值
        plane_pixel_num = h * w * plane_pixel_threshold

        max_num_plane = np.count_nonzero(face)
        instance_masks = np.zeros((8, h, w))
        face_filted = np.zeros(8)
        num_plane = 0
        for i in range(max_num_plane):
            mask = gt_seg == face[i]
            plane_pixels = np.sum(mask)
            # 如果裁剪后的图片中，某个平面像素数量占比小于阈值，则舍弃这次裁剪结果
            if plane_pixels>0:
                if plane_pixels < plane_pixel_num:
                    return None, None, None
                # if np.sum(mask)>plane_pixel_threshold:
                face_filted[num_plane] = face[i]
                instance_masks[num_plane] = mask
                num_plane = num_plane + 1
        # num_plane += 1
        instance_masks = torch.from_numpy(instance_masks)
        # print(face_filted)

        # 如果裁剪后的图片中只有一个平面，则舍弃掉这次裁剪结果
        if num_plane == 1:
            return None, None, None

        return face_filted, num_plane, instance_masks


    # intrinsix transform
    def crop_transform(self, intrinsics, old_img_size, new_img_size, h_start, w_start):
        # if using centercrop
        # old_h, old_w = old_img_size
        # new_h, new_w = new_img_size

        # intrinsics[0][2] = intrinsics[0][2] - (old_w - new_w)/2     # cx
        # intrinsics[1][2] = intrinsics[1][2] - (old_h - new_h)/2     # cy

        # if using RandomCrop
        intrinsics[0][2] = intrinsics[0][2] - w_start     # cx
        intrinsics[1][2] = intrinsics[1][2] - h_start     # cy

        return intrinsics

    def scale_transform(self, scaled_shape, cropped_shape, intrinsics):
        old_h, old_w = cropped_shape
        new_h, new_w = scaled_shape

        # print(old_h, new_h)
        # print(old_w, new_w)
        intrinsics_new = copy.deepcopy(intrinsics)
        h_ratio = new_h/old_h
        w_ratio = new_w/old_w

        intrinsics_new[0] = intrinsics[0] * w_ratio
        intrinsics_new[1] = intrinsics[1] * h_ratio

        return intrinsics_new

    def image_crop(self, image_origin, seg_origin, depth_origin):
        hw_ratio = 0.80
        img_w = random.randrange(960, 1280, 64)
        img_h = int(img_w * hw_ratio)

        # img_w = 640
        # img_h = 512
        # define transform opration
        # center_crop = albumentations.CenterCrop(p=0.8, height=img_h, width=img_w)
        rand_crop = albumentations.RandomCrop(height=img_h, width=img_w)
        # rand_crop = albumentations.RandomCrop(height=img_h, width=img_w)
        color_jitter = albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5)

        # concate 2 label at third dim
        layout_seg_ = seg_origin[:, :, np.newaxis]
        layout_depth_ = depth_origin[:, :, np.newaxis]
        masks = np.concatenate((layout_seg_, layout_depth_), 2)

        # crop_result = center_crop(image=image_origin, mask=masks)
        crop_result = rand_crop(image=image_origin, mask=masks)
        ''' the result is a dict, {'image': array, 'mask':array}
        but because i edit the source code to return h_start, w_start, so the result become {'image': (array, h_s, w_s), 'mask':(array, ~)}'''

        # print('result:', crop_result)
        cropped_image= crop_result['image'][0]

        # plt.subplot(131)
        # plt.imshow(cropped_image)
        cropped_image = color_jitter(image=cropped_image)['image']

        cropped_seg = crop_result['mask'][0][:, :, 0]
        cropped_depth = crop_result['mask'][0][:, :, 1]
        h_start, w_start = crop_result['mask'][1:]

        # plt.subplot(131)
        # plt.imshow(cropped_image)
        # plt.subplot(132)
        # plt.imshow(cropped_seg)
        # plt.subplot(133)
        # plt.imshow(cropped_depth)
        # plt.show()

        return cropped_image, cropped_seg, cropped_depth, h_start, w_start



    def get_pixel_param_maps(self, gt_seg, nd, num_face, image):
        '''
        func: cal the pixel map of n/d or abcd
        :param gt_seg: (h, w)
        :param nd: (8, 4/5),
        :param device:
        :param cfg:
        :return:
        '''
        # gt_seg = gt_seg + 1

        h, w = gt_seg.shape
        num_normal = len(nd[0])-1      # judge the is of n/d or abcd

        pixel_param_maps = np.zeros((num_normal, h, w))

        for face_id in range(num_face):
            single_plane_param = nd[face_id]
            mask = gt_seg == single_plane_param[0]
            for branch in range(num_normal):
                pixel_param_maps[branch][mask] = single_plane_param[branch+1]


        # for single_plane_normal in nd:
        #     if single_plane_normal[1] == 0:
        #         break
        #     mask = gt_seg==single_plane_normal[0]
        #     for branch in range(num_normal):
        #         pixel_param_maps[branch][mask] = single_plane_normal[branch+1]


        # plt.subplot(151)
        # plt.imshow(pixel_param_maps[0])
        # plt.subplot(152)
        # plt.imshow(pixel_param_maps[1])
        # plt.subplot(153)
        # plt.imshow(pixel_param_maps[2])
        # plt.subplot(154)
        # plt.imshow(pixel_param_maps[3])
        # plt.subplot(155)
        # plt.imshow(image)
        # plt.show()



        return pixel_param_maps

    def pick_abcd(self, face, nds):

        normal_type = len(nds[0])
        plane_normal = np.zeros((8, normal_type))


        for i, _face in enumerate(face):  # 遍历裁剪后图片的所有face
            for single_normal in nds:   # 遍历原始图片的(face,nd)
                if _face == single_normal[0]:
                    plane_normal[i] = single_normal
                    break

        return plane_normal


    def semantic_transform_face(self, face_origin, face_crop):
        '''调用这个函数的代码
        # ***tranform the semantic number in face***
            # 如果想得到转换后的face，最好先深复制一个原本的face， 因为face是一维array， 直接=复制，进行操作是会改变原本的face，进而影响后面的代码
            # face_back = copy.deepcopy(face)
            # face_trans = self.data_process.semantic_transform_face(face_origin, face_back)
        '''

        num_1 = np.count_nonzero(face_origin)
        num_2 = np.count_nonzero(face_crop)

        if 4 in face_crop:  # 裁剪后的图片包含4的情况，包括了两种face相等的情况，也就是说， 两种face相等，则face_crop总一定包含4
            return face_crop

        face_origin_ = []
        face_crop_ = []

        wall = [4, 5, 6, 7, 8]

        for i in face_origin:
            if i in wall:
                face_origin_.append(i)

        for i in face_crop:
            if i in wall:
                face_crop_.append(i)
        # 针对只截取了天花板或者地面的情况
        if len(face_crop_) == 0:
            return face_crop
        gap = face_crop_[0] - face_origin_[0]

        for i in range(num_2):
            if face_crop[i] in wall:
                face_crop[i]  = face_crop[i] - gap

        return face_crop


    def semantic_transform_seg(self, seg_origin, face_origin, face_crop):
        num_1 = np.count_nonzero(face_origin)
        num_2 = np.count_nonzero(face_crop)

        if 4 in face_crop:  # 裁剪后的图片包含4的情况，包括了两种face相等的情况，也就是说， 两种face相等，则face_crop总一定包含4
            return seg_origin

        face_origin_ = []
        face_crop_ = []

        wall = np.array([4, 5, 6, 7, 8])

        for i in face_origin:
            if i in wall:
                face_origin_.append(i)
        for i in face_crop:
            if i in wall:
                face_crop_.append(i)
        if len(face_crop_) == 0:
            return seg_origin
        gap = face_crop_[0] - face_origin_[0]

        for i in range(num_2):
            if face_crop[i] in wall:
                mask = seg_origin == face_crop[i]
                seg_origin[mask] = seg_origin[mask] - gap

        return seg_origin


    def init_trans(self, origin_init):
        mask = origin_init == 0
        origin_init[mask] = 9

        origin_init = origin_init - 1

        return origin_init

    def get_edge(self, gt_seg, bandwidth):
        '''
        using the sobel to get the gt_edge from gt_seg
        '''
        edges =  filters.sobel(gt_seg)
        kernel = np.ones((2, 2), np.uint8)
        gt_edge = cv2.dilate(edges, kernel, iterations=bandwidth)
        mask = gt_edge > 0.0001
        gt_edge[mask] = 1

        # plt.imshow(gt_edge)
        # plt.show()

        return gt_edge

    def label_trans(self,gt_seg, face, instance_plane_params):
        '''  trans the label of 3~7 to 34576'''
        face = [i for i in face if i>0]
        face_cf = [i - 1 for i in face if (i<4) and (i>0) ]      # (1, 3) -->  (0,2)
        face_wall = [i - 2 for i in face if i > 3]               # (4,5,6,7,8,9) -- >(2, 3,4,5,6,7)
        num_wall = len(face_wall)

        trans_seg = np.zeros_like(gt_seg)

        for itr, single_face in enumerate(face_cf):     # (0,2) --> (0,1)
            if single_face == 2:
                face_cf[int(itr)] = 1

        if num_wall == 1:
            face_wall_new = [2]
        elif num_wall == 2:
            face_wall_new = [2, 3]
        elif num_wall == 3:
            face_wall_new = [2, 4, 3]
        elif num_wall == 4:
            face_wall_new = [2, 4, 5, 3]
        elif num_wall == 5:
            face_wall_new = [2, 4, 6, 5, 3]
        # elif num_wall == 6:
        #     face_wall_new = [2, 4, 6, 7, 5, 3]
        else:
            print('wrong, got picture of wall more than six !!!')

        trans_face = np.array(face_cf + face_wall_new)

        # face_filled = np.zeros(9)
        face_filled = np.array([10, 10, 10 ,10, 10, 10, 10, 10, 10])
        for index, single_face in enumerate(face):
            wall_mask = gt_seg == single_face
            # print(111, single_face)
            # print(222, np.count_nonzero(wall_mask))
            trans_seg[wall_mask] = trans_face[index]
            face_filled[index] = trans_face[index]

        instance_plane_params[:, 0] = face_filled.reshape(9)


        return face_filled, instance_plane_params, trans_seg

    def label_trans_one_side(self, gt_seg, face, instance_plane_params):
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

