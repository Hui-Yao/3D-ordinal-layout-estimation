import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from skimage import filters

class TotalLoss():
    def __init__(self, cfg, device, K):
        self.h = cfg.dataset.h
        self.w = cfg.dataset.w
        self.num_class = cfg.dataset.num_class
        self.batch_size = cfg.dataset.batch_size
        self.device = device
        self.K = K
        self.k_inv_dot_uv = self.K_inv_dot_uv()     # 初始化函数中用到的类属性要在之前定义

    def K_inv_dot_uv(self):
        h, w = self.h, self.w
        K = self.K
        device = self.device

        K_inv = np.linalg.inv(np.array(K))
        K_inv = torch.FloatTensor(K_inv).to(device)

        x = torch.arange(w, dtype=torch.float32).view(1, w) / w * 640
        y = torch.arange(h, dtype=torch.float32).view(h, 1) / h * 480
        x = x.to(device)
        y = y.to(device)
        xx = x.repeat(h, 1)  # 将x沿轴0把元素复制h倍，得到xx.shape = (192, 256)  1表示哪个轴不翻倍
        yy = y.repeat(1, w)  # 将y沿轴1把元素复制w倍，得到yy.shape = (192, 256)
        xy1 = torch.stack((xx, yy, torch.ones((h, w), dtype=torch.float32).to(device)))  # (3, h, w)
        xy1 = xy1.view(3, -1)  # (3, h*w) , 将上面的3维Tensor填充一个(3, h*w)的tensor
        k_inv_dot_xy1 = torch.matmul(K_inv, xy1)  # (3, h*w) = (3,3) * (3, hw)， 将相机内参和放大图相乘， 得到coordinate_map

        return k_inv_dot_xy1

    def mse_loss(self, pred_prob_8, gt_seg):    # pred_prob_8：所有类别通道上的概率， gt_seg需要转化为onehot， 也即是希望某个像素在其类别通道上的概率要为1
        h, w = self.h, self.w
        pred_prob = pred_prob_8.view(-1, h*w)
        gt_onehot = nn.functional.one_hot(gt_seg, num_classes=self.num_class).float()
        # plt.subplot(241)
        # plt.imshow(gt_onehot[:, :, 0].cpu().numpy())
        # plt.subplot(242)
        # plt.imshow(gt_onehot[:, :, 1].cpu().numpy())
        # plt.subplot(243)
        # plt.imshow(gt_onehot[:, :, 2].cpu().numpy())
        # plt.subplot(244)
        # plt.imshow(gt_onehot[:, :, 3].cpu().numpy())
        # plt.subplot(245)
        # plt.imshow(gt_onehot[:, :, 4].cpu().numpy())
        # plt.subplot(246)
        # plt.imshow(gt_onehot[:, :, 5].cpu().numpy())
        # plt.subplot(247)
        # plt.imshow(gt_onehot[:, :, 6].cpu().numpy())
        # plt.subplot(248)
        # plt.imshow(gt_onehot[:, :, 7].cpu().numpy())
        # plt.show()
        gt_onehot = gt_onehot.view(h*w, -1).t()
        mse = F.mse_loss(pred_prob, gt_onehot)
        return mse

    def ordinal_ce(self, pred_feature, gt_seg, batch_weight, class_weight = True, ordinal_weight = True):
        def cal_order_weight(pred_feature, gt_seg):
            pred_seg = torch.max(pred_feature)[0]
            order_weight = torch.abs(pred_seg - gt_seg)
            order_weight += 1

            return order_weight

        if class_weight and not ordinal_weight:
            _loss = F.cross_entropy(pred_feature, gt_seg, weight=batch_weight)
        elif not class_weight and ordinal_weight:
            order_weight = cal_order_weight(pred_feature, gt_seg)
            _loss = F.cross_entropy(pred_feature, gt_seg, reduction='none')
            _loss = torch.mean(_loss * order_weight)
        elif class_weight and ordinal_weight:
            order_weight = cal_order_weight(pred_feature, gt_seg)
            _loss = F.cross_entropy(pred_feature, gt_seg, weight=batch_weight, reduction='none')
            _loss = torch.mean(_loss * order_weight)
        else:
            _loss = F.cross_entropy(pred_feature, gt_seg, weight=batch_weight)

        return _loss

    def multi_dice_loss(self, input, target, weight=None):
        '''
        input: pred_prob_8, shape=(1, 8, h, w)
        target: shape = (1, 192, 256)
        '''
        B, C, h, w = input.shape

        target = nn.functional.one_hot(target.view(h*w), num_classes=self.num_class).float()
        input = input.view(C, h*w).T

        totalLoss = 0

        for i in range(C):
            diceLoss = self.dice_loss(input[:, i], target[:, i])
            if weight is not None:
                diceLoss *= weight[i]
            totalLoss += diceLoss

        return totalLoss


    def surface_normal_loss(self, prediction, gt_normal):  # pred_param:(4 h, w), gt_param: (4 h, w)
        c = prediction.shape[0]
        prediction = prediction.view(c, -1).t()
        gt_normal = gt_normal.view(c, -1).t()

        similarity = torch.nn.functional.cosine_similarity(prediction, gt_normal, dim=1)  # shape = (hw)
        # 分别计算3个参数n/d的相似度，得到shape=(3)的tensor
        loss = torch.mean(1-similarity)     # 计算平均误差
        mean_angle = torch.mean(torch.acos(torch.clamp(similarity, -1, 1)))
        return loss, mean_angle / np.pi * 180
        # torch.clamp(t, low, high):将张量t中的值truncate到[low, high]之间.truncate是指小于low或大于high的值直接变为low和high，处于中间的值则不变
        # torch.acos(t)：求取余弦值对应的角度(0-pi)。返回的是与t相同形状的数据。注意：t中元素要在[-1, 1]之间

    # L1 parameter loss
    def parameter_loss(self, prediction, param):
        c, h, w = prediction.size()
        valid_predition = torch.transpose(prediction.view(c, -1), 0, 1)
        valid_param = torch.transpose(param.view(c, -1), 0, 1)

        return torch.mean(torch.sum(torch.abs(valid_predition - valid_param), dim=1))

    def Q_loss(self, pred_normald, gt_depth):
        '''
        infer per pixel depth using perpixel plane parameter and
        return depth loss, mean abs distance to gt depth, perpixel depth map
        :param param: plane parameters defined as n/d , tensor with size (1, 3, h, w)
        :param k_inv_dot_xy1: tensor with size (3, h*w)
        :param depth: tensor with size(1, 1, h, w)
        :return: error and abs distance
        '''

        c, h, w = pred_normald.size()

        k_inv_dot_uv = self.k_inv_dot_uv
        gt_depth = gt_depth.view(1, h * w)
        pred_normald = pred_normald.view(c, h * w)

        # infer depth for every pixel
        inferred_depth = 1. / torch.sum(pred_normald * k_inv_dot_uv, dim=0, keepdim=True).view(1, -1)  # torch.sum((3, hw), dim=0)-->(1, h*w)
        inferred_depth = torch.clamp(inferred_depth, 1e-4, 15.0)

        diff = torch.abs(inferred_depth - gt_depth)  # gt_depth与infered_depth对应像素级的差值组成的mask。对平面/非平面都进行监督
        abs_distance = torch.mean(diff)  # 每个像素的平均深度差

        Q = k_inv_dot_uv * gt_depth  # (3, n)
        # q_diff = torch.abs(torch.sum(pred_abc * Q, dim=0, keepdim=True) - pred_d)  # 也是像素级的监督，nT Q = 1
        q_diff = torch.abs(torch.sum(pred_normald * Q, dim=0, keepdim=True) - 1.)  # 也是像素级的监督，nT Q = 1
        loss = torch.mean(q_diff)  #

        # return loss, abs_distance, infered_depth.view(1, 1, h, w)
        return loss, abs_distance, inferred_depth

    def calc_frustum_planes(self):
        '''
        Calculate frustum planes
        :return: A list of frustum planes
        '''

        h, w = self.h, self.w
        K = self.K
        device = self.device

        K_inv = np.linalg.inv(K)
        c1 = np.array([0, 0, 1])      # c3------c4
        c2 = np.array([w - 1, 0, 1])  # |        |
        c3 = np.array([0, h - 1, 1])  # c1------c2
        c4 = np.array([w - 1, h - 1, 1])

        v1 = K_inv.dot(c1)
        v2 = K_inv.dot(c2)
        v3 = K_inv.dot(c3)
        v4 = K_inv.dot(c4)
        n12 = np.cross(v1, v2)
        n12 = n12 / np.sqrt(n12[0] ** 2 + n12[1] ** 2 + n12[2] ** 2)
        n13 = -np.cross(v1, v3)
        n13 = n13 / np.sqrt(n13[0] ** 2 + n13[1] ** 2 + n13[2] ** 2)
        n24 = -np.cross(v2, v4)
        n24 = n24 / np.sqrt(n24[0] ** 2 + n24[1] ** 2 + n24[2] ** 2)
        n34 = -np.cross(v3, v4)
        n34 = n34 / np.sqrt(n34[0] ** 2 + n34[1] ** 2 + n34[2] ** 2)
        plane1 = np.concatenate((n12, [0]))
        plane2 = np.concatenate((n13, [0]))
        plane3 = np.concatenate((n24, [0]))
        plane4 = np.concatenate((n34, [0]))
        frustum_planes = [plane1, plane2, plane3, plane4]
        frustum_planes = torch.tensor(frustum_planes).to(device)

        return frustum_planes

    def instance_parameter_loss(self, pred_segmentation, pred_params, gt_depth, return_loss=True):
        # 预测的实例分割图； 抽样点的实力分割图； 抽样点的n/d；平面区域； GT深度图
        """ N表示抽样点数量， K表示预测的center的数量，也就是图片中的平面数量
        calculate loss of parameters
        first we combine sample segmentation with sample params to get K plane parameters
        then we used this parameter to infer plane based Q loss as done in PlaneRecover
        the loss enforce parameter is consistent with ground truth depth

        :param segmentation: tensor with size (h*w, K)  预测的实例分割图；
        :param sample_segmentation: tensor with size (N, K) 抽样点的实力分割图；
        :param sample_params: tensor with size (3, N), defined as n / d  抽样点的n/d；
        :param valid_region: tensor with size (1, 1, h, w), indicate planar region  平面区域；
        :param gt_depth: tensor with size (1, 1, h, w)  GT深度图
        :param return_loss: bool
        :return: loss
                 infered depth with size (1, 1, h, w) corresponded to instance parameters
        """
        h, w = self.h, self.w
        k_inv_dot_uv = self.k_inv_dot_uv

        # combine sample segmentation and sample params to get instance parameters
        # if not return_loss:
        #     pred_segmentation[pred_segmentation < 0.5] = 0.  # 将预测概率比0小的值设为0,元素总数还是N个

        pred_segmentation = pred_segmentation.view(-1, h*w).T   # (hw, 8)
        pred_params = pred_params.view(-1, h*w)                 # (4, hw)
        # (N, k) --> (N, k) , 对一个像素在k个通道上的概率做规范化，相当于乘以的论文中公式前的归一化系数,使得所有通道上的数值都处于0~1之间
        weight_matrix = F.normalize(pred_segmentation, p=1, dim=0)  # (hw, 8)

        instance_param = torch.matmul(pred_params, weight_matrix)  # (4, K) = (4, hw) * (hw, K) , k个平面的实例级参数
        # instance_param = instance_param1[0:3, :] / instance_param1[3, :]

        # infer depth for every pixels and select the one with highest probability
        depth_maps = 1. / torch.matmul(instance_param.t(), k_inv_dot_uv)  # (K, h*w) = (k, 3)*(3, h*w), 相当于每个平面都覆盖整个房间
        _, index = pred_segmentation.max(dim=1)  # (hw, k)-->(hw), 总之就是在通道数上求最大值，也就是这个像素的类别
        inferred_depth = depth_maps.t()[range(h * w), index].view(1, -1)  # (hw), 从多张深度图取得对应区域组成inferred深度图
        inferred_depth = torch.clamp(inferred_depth, 1e-4, 15.0)
        # 注意这里是range， 对于每个像素的k通道，根据index在通道内取出对应的值，每个像素取一个值，则(hw,k)——>(hw)
        # plt.imshow(inferred_depth.cpu().detach().numpy().reshape(h, w))
        # plt.show()
        # Q_loss for every instance
        gt_depth = gt_depth.view(1, -1)
        Q = gt_depth * k_inv_dot_uv  # Z(X/Z, Y/Z, 1) = (X, Y, Z),求得三维点, shape of (3, N)

        # (k, 3) * (3, hw) = (k, hw), 那么每个hw表示的就是这个布局平面扩展到整个图片大小，但是不过这个平面多大，这个平面上每一点的参数中offset都是相同的
        Q_loss = torch.abs(torch.matmul(instance_param.t(), Q) - 1.)
        # Q_loss = torch.abs(torch.matmul(pred_nd.t(), Q) - 1)   # (K, N)， 三维点和平面参数乘积，求loss

        # weight Q_loss with probability  ，在每个像素的loss前加个权重
        weighted_Q_loss = Q_loss * pred_segmentation.t()  # (K, N)， 只对有效区域的像素求loss
        loss = torch.sum(torch.mean(weighted_Q_loss, dim=1))  # (K)——>scalar，对图片中所有平面的loss求和，得到一张图片的loss

        abs_distance = torch.mean(torch.abs(inferred_depth - gt_depth))  # 像素级平面参数得到的pred depthmap与gt depthmap之间的差距

        return loss, inferred_depth, abs_distance

