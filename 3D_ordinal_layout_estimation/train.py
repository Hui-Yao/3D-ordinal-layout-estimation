'''
time:2021.4.16
'''
# official package
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as tf

import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import albumentations
import cv2
from easydict import EasyDict as edict

import socket
import copy
import random
import os
import yaml
import time
import json

# own
from model.baseline_same import Baseline as UNet
from utils.misc import get_optimizer, adjust_learning_rate
from utils.metrics import SegmentationMetric, DepthEstmationMetric
from utils.dataset_process import DatasetProcess
from utils.layout_loss import TotalLoss
from utils.layout_plot import layout_plot
from utils.misc import cal_batch_weight


class PlaneDataset(data.Dataset):  # PlaneDataset(subset=subset, transform=transforms, root_dir=cfg.root_dir)
    def __init__(self, subset='train', transform=None, root_dir=None, img_h=192, img_w=256):
        self.h = img_h
        self.w = img_w
        self.bandwidth = 1
        self.subset = subset
        self.transform = transform
        self.color_jitter = albumentations.ColorJitter()
        self.data_process = DatasetProcess(self.h, self.w)

        self.data_dir = os.path.join(root_dir, f'{subset}')
        record_path = os.path.join(self.data_dir, f'interiornet_layout_{subset}.npy')
        self.image_name_list = np.load(record_path)

    def __getitem__(self, index):
        h, w = self.h, self.w
        jpg_name = self.image_name_list[index] + '.jpg'  # for image of input
        png_name = self.image_name_list[index] + '.png'  # for image of label
        npy_name = self.image_name_list[index] + '.npy'  # for npy file of label

        img_path = os.path.join(self.data_dir, 'image', jpg_name)  # 根据index，确定数据路径
        layout_seg_path = os.path.join(self.data_dir, 'layout_seg', png_name)  # 根据index，确定数据路径
        layout_depth_path = os.path.join(self.data_dir, 'layout_depth', png_name)  # 根据index，确定数据路径
        face_path = os.path.join(self.data_dir, 'face', npy_name)
        layout_keypoint_path = os.path.join(self.data_dir, 'layout_keypoint', npy_name)
        plane_params4_path = os.path.join(self.data_dir, 'plane_params4', npy_name)
        plane_params5_path = os.path.join(self.data_dir, 'plane_params5', npy_name)

        image = io.imread(img_path)
        # layout_seg = io.imread(layout_seg_path).astype(np.uint8)
        layout_seg = io.imread(layout_seg_path)
        layout_depth = io.imread(layout_depth_path) / 4000
        face = np.load(face_path)
        face_exclude_zero = [i for i in face if i >0]
        num_face = len(face_exclude_zero)
        layout_keypoint = np.load(layout_keypoint_path)
        # instance_plane_params = np.load(plane_params4_path)
        instance_plane_params = np.load(plane_params4_path)
        intrinsics_matrix = np.array([[600, 0, 320],
                                      [0, 600, 240],
                                      [0, 0, 1]])

        image = cv2.resize(image, (w, h))
        face, instance_plane_params, layout_seg = self.data_process.label_trans_one_side(layout_seg, face,instance_plane_params)  # trans the label from 3~7 to 34576
        layout_seg = cv2.resize(layout_seg, (w, h))
        layout_depth = cv2.resize(layout_depth, (w, h))

        pixel_param_maps = self.data_process.get_pixel_param_maps(layout_seg, instance_plane_params, num_face, image)

        if self.subset == 'train':
            image = self.color_jitter(image=image)['image']
        if self.transform is not None:
            image = self.transform(image)

        sample = {
            'image': image,  # type = ndarray, shape = (h, w, 3)
            'gt_layout_seg': layout_seg,  # type = ndarray, shape = (h, w)
            'gt_layout_depth': layout_depth,  # type = ndarray, shape = (h, w)
            'intrinsic': intrinsics_matrix,
            'gt_pixel_param_maps': pixel_param_maps,
            'gt_layout_keypoint': layout_keypoint
        }

        return sample

    def __len__(self):
        return len(self.image_name_list)


def load_dataset(subset, cfg):  # 'trian/test' cfg.dataset
    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    is_shuffle = subset == 'train'  # 只有train才shuffle
    loaders = data.DataLoader(
        PlaneDataset(subset=subset, transform=transforms, root_dir=cfg.root_dir, img_h=cfg.h, img_w=cfg.w),
        batch_size=cfg.batch_size, shuffle=is_shuffle, num_workers=cfg.num_workers, drop_last=True)

    return loaders


def training(cfg):
    # 给三个包中的随机函数设置种子
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型保存路径
    checkpoint_dir = os.path.join(cfg.checkpoint_dir, cfg.model_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 保存训练参数
    hyper_record_path = os.path.join(cfg.checkpoint_dir, '00_hyper_record.txt')
    with open(hyper_record_path, 'w') as f:
        json_str = json.dumps(cfg, indent=0)
        f.write(json_str)
        f.write('\n')

    # build network    # from models.baseline_same import Baseline as UNet
    network = UNet(cfg.model)   # resnet101-FPN

    # load nets into gpu    多GPU分布式
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        network = torch.nn.DataParallel(network)
    network.to(device)

    # set up optimizers  建立优化器
    optimizer = get_optimizer(network.parameters(), cfg.solver)

    data_loader = load_dataset('train', cfg.dataset)

    network.train()

    # save losses per epoch
    history = {'losses_plane_ce': [], 'losses_plane_dice': [],
               'losses_normal': [], 'losses_L1': [], 'losses_depth': [], 'losses_instance': []}
    history_val = {'losses_plane_ce': [], 'losses_plane_dice': [],
                   'losses_normal': [], 'losses_L1': [], 'losses_depth': [], 'losses_instance': []}

    metric = {'pa': [], 'mpa': [], 'cpa': [], 'miou': [], 'ciou': [],
              'rmse': [], 'log10': [], 'abs_rel': [], 'sq_rel': [], 'accu0': [], 'accu1': [], 'accu2': []}
    metric_val = {'pa': [], 'mpa': [], 'cpa': [], 'miou': [], 'ciou': [],
                  'rmse': [], 'log10': [], 'abs_rel': [], 'sq_rel': [], 'accu0': [], 'accu1': [], 'accu2': []}

    best_accu = {'mpa': 0, 'miou': 0, 'accu0': 0,
                 'plane_str1:': '0', 'depth_str1': '0',
                 'plane_str2:': '0', 'depth_str2': '0',
                 'plane_str3:': '0', 'depth_str3': '0'}

    for epoch in range(cfg.num_epochs):
        # reset the picture conter every epoch
        plane_metric = SegmentationMetric(7)
        depth_metric = DepthEstmationMetric(cfg)

        adjust_learning_rate(cfg, optimizer, epoch)  # 动态调整学习率

        losses_plane_ce, losses_plane_dice, losses_normal, losses_L1, losses_instance, losses_depth= \
            0., 0., 0., 0., 0., 0.
        num_batch = 0   # 统计每个epoch中有多少个batch

        for iter, sample in enumerate(data_loader):  # 每次取出batch_size个数据
            image = sample['image'].to(device).float()  # (bs, 3, h, w)     RGB三通道图片
            gt_seg = sample['gt_layout_seg'].to(device).long()  # (bs, h, w)    # 布局平面Gt语义分割图
            gt_depth = sample['gt_layout_depth'].to(device).float()  # (bs, h, w)    # 布局平面Gt语义分割图
            intrinsic = sample['intrinsic'].float()  # shape = (bs, 3, 3), cpu
            gt_pixel_param_maps = sample['gt_pixel_param_maps'].to(device).float()

            # forward pass
            pred_feature_8, pred_prob_8, pred_class_1, pred_surface_normal_4 = network(image)
            loss, loss_plane_ce, loss_plane_dice, loss_normal, loss_L1, loss_depth, loss_instance, loss_keypoint = \
                0., 0., 0., 0., 0., 0., 0., 0.

            batch_weight = cal_batch_weight(cfg.dataset.num_class, gt_seg)      # cal the class weight of a batch data

            batch_size = cfg.dataset.batch_size
            for i in range(batch_size):
                total_loss = TotalLoss(cfg, device, K=intrinsic[i])

                _loss_plane_ce = total_loss.ordinal_ce(pred_feature_8, gt_seg, batch_weight, class_weight=True, ordinal_weight=False)

                # plane param loss
                # cosine smilarity
                _loss_normal, mean_angle = total_loss.surface_normal_loss(pred_surface_normal_4[i], gt_pixel_param_maps[i])
                # # L1 loss
                _loss_L1 = total_loss.parameter_loss(pred_surface_normal_4[i], gt_pixel_param_maps[i])
                # # Q_loss
                _loss_depth, _loss_distance, inferred_pixel_depth = total_loss.Q_loss(pred_surface_normal_4[i], gt_depth[i])
                #
                # # instance pooling
                # _loss_instance = 0
                _loss_instance, inferred_plane_depth, abs_distance = \
                    total_loss.instance_parameter_loss(pred_prob_8[i], pred_surface_normal_4[i], gt_depth[i])

                # total loss
                _loss = _loss_plane_ce + _loss_normal + _loss_L1 + _loss_depth + _loss_instance

                with torch.no_grad():
                    pa, cpa, mpa, ciou, miou = plane_metric(pred_class_1, gt_seg[i])
                    rmse, log10, abs_rel, sq_rel, accu0, accu1, accu2 = depth_metric(inferred_plane_depth, gt_depth[i])

                loss += _loss
                loss_plane_ce += _loss_plane_ce

                loss_normal += _loss_normal
                loss_L1 += _loss_L1
                loss_depth += _loss_depth
                loss_instance += _loss_instance

                # 统计一个batch每张图片的平均loss
            loss = loss / batch_size
            loss_plane_ce /= batch_size
            loss_plane_dice /= batch_size

            loss_normal /= batch_size
            loss_L1 /= batch_size
            loss_depth /= batch_size
            loss_instance /= batch_size

            # Backward
            # 在每一batch计算中，优化器梯度空间清零
            optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()

            # 一个batch中每张图片的平均损失 × batch数量， 也就是一个epoch中所有图片的损失 ÷ batch_size
            losses_plane_ce += loss_plane_ce.item()
            losses_plane_dice += loss_plane_dice

            losses_normal += loss_normal.item()
            losses_L1 += loss_L1.item()
            losses_depth += loss_depth.item()
            losses_instance += loss_instance.item()

            num_batch += 1  # 统计一个epoch中有多少个epoch

            if num_batch%10 == 0:
                print('{time}, processing epoch_{epoch}, batch_{num_batch}'
                      .format(time = time.strftime('%Y-%m-%d, %H:%M:%S'), epoch = epoch, num_batch = num_batch))

        # save history  # 计算每个epoch中每张图片的平均的损失
        # 因为epoch = batch_size * num_batch, 所以 一个epoch中每张图片的平均损失 = 一个epoch图片总损失/(batch_size * num_batch)
        history['losses_plane_ce'].append(losses_plane_ce / num_batch)
        history['losses_plane_dice'].append(losses_plane_dice / num_batch)

        history['losses_normal'].append(losses_normal / num_batch)
        history['losses_L1'].append(losses_L1 / num_batch)
        history['losses_depth'].append(losses_depth / num_batch)
        history['losses_instance'].append(losses_instance / num_batch)

        metric['pa'].append(pa)
        metric['cpa'].append(cpa)
        metric['mpa'].append(mpa)
        metric['ciou'].append(ciou)
        metric['miou'].append(miou)

        metric['rmse'].append(rmse)
        metric['log10'].append(log10)
        metric['abs_rel'].append(abs_rel)
        metric['sq_rel'].append(sq_rel)
        metric['accu0'].append(accu0)
        metric['accu1'].append(accu1)
        metric['accu2'].append(accu2)

        print('*' * 100)
        print(f'just processed epoch_{epoch}:')
        print('losses_plane_ce:', losses_plane_ce / num_batch)
        print('losses_plane_dice:', losses_plane_dice / num_batch)

        print('losses_normal:', losses_normal / num_batch)
        print('losses_L1:', losses_L1 / num_batch)
        print('losses_depth:', losses_depth / num_batch)
        print('losses_instance:', losses_instance / num_batch)
        print('~' * 60)
        print('rmse: ', rmse)
        print('log10: ', log10)
        print('abs_rel:', abs_rel)
        print('sq_rel: ', sq_rel)
        print('accu0: ', accu0)
        print('accu1: ', accu1)
        print('accu2: ', accu2)
        print('*' * 100)

        network_save_path = os.path.join(checkpoint_dir, 'network')
        if not os.path.exists(network_save_path):
            os.makedirs(network_save_path)
        history_save_path = os.path.join(checkpoint_dir, 'history_metric')
        if not os.path.exists(history_save_path):
            os.makedirs(history_save_path)

        # save model per epoch
        torch.save(network.state_dict(), os.path.join(network_save_path, f"network_epoch_{epoch}.pt"))

        cfg_val = copy.deepcopy(cfg)
        history_val = validation(cfg_val, history_val, metric_val, best_accu, epoch, device)

        if (epoch + 1) % 4 == 0:
            # if True:
            layout_plot(history, history_val, metric, metric_val, checkpoint_dir, epoch)
        # save history and metric per epoch
        np.savez(os.path.join(history_save_path, f'history_metric_{epoch}.npz'),
                 history_train=history, history_val=history_val, metric_train=metric, metric_val=metric_val)
        # pickle.dump(history, open(os.path.join(checkpoint_dir, 'history_epoch_%d.pkl' % epoch), 'wb'))
        # pickle.dump(history_val, open(os.path.join(checkpoint_dir, 'history_val_epoch_%d.pkl' % epoch), 'wb'))


def validation(cfg, history_val, metric_val, best_accu, epoch, device):
    # 给三个包中的随机函数设置种子
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    cfg.dataset.batch_size = 1

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #创建模型保存路径
    checkpoint_dir = cfg.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # build network    # from models.baseline_same import Baseline as UNet
    network = UNet(cfg.model)  # resnet101-FPN

    if not (cfg.resume_dir == 'None'):  # 只有eval()需要resume
        resume_dir = cfg.resume_dir + '/network' + '/network_epoch_%d.pt'%epoch
        model_dict = torch.load(resume_dir, map_location=lambda storage, loc: storage)
        network.load_state_dict({k.replace('module.', ''): v for k, v in model_dict.items()})   # sever
        # network.load_state_dict(model_dict)     # my computer

    # load nets into gpu    多GPU分布式
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        network = torch.nn.DataParallel(network)
    network.to(device)  #
    network.eval()

    # data loader
    data_loader = load_dataset('val', cfg.dataset)

    plane_metric = SegmentationMetric(7)
    depth_metric = DepthEstmationMetric(cfg)

    with torch.no_grad():
        losses_plane_ce, losses_plane_dice, losses_normal, losses_L1, losses_instance, losses_depth, losses_keypoint = \
            0., 0., 0., 0., 0., 0., 0.

        num_batch = 0  # 统计每个epoch中有多少个batch
        for iter, sample in enumerate(data_loader):  # 每次取出batch_size个数据
            image = sample['image'].to(device).float()  # (bs, 3, h, w)     RGB三通道图片
            gt_seg = sample['gt_layout_seg'].to(device).long()  # (bs, h, w)    # 布局平面Gt语义分割图
            gt_depth = sample['gt_layout_depth'].to(device).float()  # (bs, h, w)    # 布局平面Gt语义分割图
            intrinsic = sample['intrinsic'].float()  # shape = (bs, 3, 3), cpu
            gt_pixel_param_maps = sample['gt_pixel_param_maps'].to(device).float()
            # gt_keypoints = sample['gt_keypoints']

            # forward pass
            pred_feature_8, pred_prob_8, pred_class_1, pred_surface_normal_4 = network(image)

            batch_size = cfg.dataset.batch_size
            for i in range(batch_size):
                total_loss = TotalLoss(cfg, device, K=intrinsic[i])

                _loss_plane_ce = F.cross_entropy(pred_feature_8[i].unsqueeze(0), gt_seg[i].unsqueeze(0))  # CE中会对输入特征图做softmax

                _loss_plane_dice = total_loss.multi_dice_loss(pred_prob_8[i].unsqueeze(0), gt_seg[i].unsqueeze(0))

                # plane param loss
                # cosine smilarity
                _loss_normal, mean_angle = total_loss.surface_normal_loss(pred_surface_normal_4[i], gt_pixel_param_maps[i])
                # # L1 loss
                _loss_L1 = total_loss.parameter_loss(pred_surface_normal_4[i], gt_pixel_param_maps[i])
                # # Q_loss
                _loss_depth, _loss_distance, inferred_pixel_depth = total_loss.Q_loss(pred_surface_normal_4[i], gt_depth[i])

                # # instance pooling
                # _loss_instance = 0
                _loss_instance, inferred_plane_depth, abs_distance, plane_list = \
                    total_loss.instance_parameter_loss(pred_prob_8[i], pred_surface_normal_4[i], gt_depth[i])

                # total loss
                _loss = _loss_plane_ce + 0.1*_loss_plane_dice + _loss_normal + _loss_L1 + _loss_depth + _loss_instance

                pa, cpa, mpa, ciou, miou = plane_metric(pred_class_1, gt_seg)
                rmse, log10, abs_rel, sq_rel, accu0, accu1, accu2 = depth_metric(inferred_plane_depth, gt_depth[i])

            # 一个batch中每张图片的平均损失 × batch数量， 也就是一个epoch中所有图片的损失 ÷ batch_size
            losses_plane_ce += _loss_plane_ce
            losses_plane_dice += _loss_plane_dice

            losses_normal += _loss_normal
            losses_L1 += _loss_L1
            losses_depth += _loss_depth
            losses_instance += _loss_instance

            num_batch += 1  # 统计一个epoch中有多少个epoch
            if num_batch%200 == 0:
                print('{}, processing the val dataset, epoch_{}, picture_{}'.format(time.strftime('%Y-%m-%d, %H:%M:%S'), epoch, num_batch))

        history_val['losses_plane_ce'].append(losses_plane_ce / num_batch)
        history_val['losses_plane_dice'].append(losses_plane_dice / num_batch)

        history_val['losses_normal'].append(losses_normal / num_batch)
        history_val['losses_L1'].append(losses_L1 / num_batch)
        history_val['losses_depth'].append(losses_depth / num_batch)
        history_val['losses_instance'].append(losses_instance / num_batch)

        metric_val['pa'].append(pa)
        metric_val['cpa'].append(cpa)
        metric_val['mpa'].append(mpa)
        metric_val['ciou'].append(ciou)
        metric_val['miou'].append(miou)

        metric_val['rmse'].append(rmse)
        metric_val['log10'].append(log10)
        metric_val['abs_rel'].append(abs_rel)
        metric_val['sq_rel'].append(sq_rel)
        metric_val['accu0'].append(accu0)
        metric_val['accu1'].append(accu1)
        metric_val['accu2'].append(accu2)

        best_record_path = os.path.join(cfg.checkpoint_dir, '001_best_record.txt')

        plane_metric_str = 'epoch: {z}\npa: {a}\nmpa: {c}\ncpa: {b}\nmoiu: {e}\nciou: {d}\n\n'.\
            format(z=epoch, a=pa,b=cpa, c=mpa,d=ciou, e=miou)
        depth_metric_str = 'epoch: {z}\nrmse: {a}\nlog10: {b}\nabs_rel: {c}\nsq_rel: {d}\naccu0: {e}\naccu1: {f}\naccu2: {g}'. \
            format(z=epoch, a=rmse, b=log10, c=abs_rel, d=sq_rel, e=accu0, f=accu1, g=accu2)

        if mpa > best_accu['mpa']:
            best_accu['mpa'] = mpa
            best_accu['plane_str1'] = plane_metric_str
            best_accu['depth_str1'] = depth_metric_str
        if miou > best_accu['miou']:
            best_accu['miou'] = miou
            best_accu['plane_str2'] = plane_metric_str
            best_accu['depth_str2'] = depth_metric_str
        if accu0 > best_accu['accu0']:
            best_accu['accu0'] = accu0
            best_accu['plane_str3'] = plane_metric_str
            best_accu['depth_str3'] = depth_metric_str

        with open(best_record_path, 'w') as f:
            # f.write('\nepoch:{a}\n\n'.format(a=epoch))
            f.write(f'\n~~~~~~~~~~~~~~~~~~best performance for plane mpa~~~~~~~~~~~~~~~~~~\n')
            f.write(best_accu['plane_str1'])
            f.write('\n')
            f.write(best_accu['depth_str1'])
            f.write(f'\n~~~~~~~~~~~~~~~~~~best performance for plane miou~~~~~~~~~~~~~~~~~~\n')
            f.write(best_accu['plane_str2'])
            f.write('\n')
            f.write(best_accu['depth_str2'])
            f.write(f'\n~~~~~~~~~~~~~~~~~~best performance for depth estimation~~~~~~~~~~~~~~~~~~\n')
            f.write(best_accu['plane_str3'])
            f.write('\n')
            f.write(best_accu['depth_str3'])
            # f.write('*'*80)

        print('*' * 100)
        print(f'just processed epoch_{epoch}:')
        print('losses_plane_ce:', losses_plane_ce / num_batch)
        print('losses_plane_dice:', losses_plane_dice / num_batch)

        print('losses_normal:', losses_normal / num_batch)
        print('losses_L1:', losses_L1 / num_batch)
        print('losses_depth:', losses_depth / num_batch)
        print('losses_instance:', losses_instance / num_batch)
        print('*' * 100)

        print('pa:', pa)
        print('cpa:', cpa)
        print('mpa:', mpa)
        print('ciou:', ciou)
        print('miou:', miou)

        print('~' * 60)
        print('rmse: ', rmse)
        print('log10: ', log10)
        print('abs_rel:', abs_rel)
        print('sq_rel: ', sq_rel)
        print('accu0: ', accu0)
        print('accu1: ', accu1)
        print('accu2: ', accu2)
        print('#' * 100)

    return history_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--model_name', type=str, help='the name of new model')
    arg = parser.parse_args()

    yaml_dir = 'config/layout_config.yaml'
    with open(yaml_dir) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        cfg = edict(config)

    cfg.dataset.root_dir = arg.data_path
    cfg.model_name = arg.model_name

    training(cfg)



