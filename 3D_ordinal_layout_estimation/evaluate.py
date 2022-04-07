# official package
import torch
import torchvision.transforms as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.io as io
from easydict import EasyDict as edict
import yaml
import os
import argparse
import copy
import glob

# own
from model.baseline_same import Baseline as UNet
from depth_intersection_module.depth_intersection import DepthIntersection
from utils.metrics import SegmentationMetric, DepthEstmationMetric
from utils.cal_3d_corner_error import corner_error


parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--data_path', type=str, help='path to testing set')
parser.add_argument('--pretrained_path', type=str, help='path to pretrained model')
arg = parser.parse_args()

yaml_dir = 'config/layout_config.yaml'
with open(yaml_dir) as f:
    config = yaml.load(f,Loader=yaml.FullLoader)
    cfg = edict(config)

cfg.update(vars(arg))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = UNet(cfg.model)  # resnet101-FPN

resume_dir = cfg.pretrained_path
model_dict = torch.load(resume_dir, map_location=lambda storage, loc: storage)
network.load_state_dict(model_dict)

network.to(device)
network.eval()
h, w = cfg.dataset.h, cfg.dataset.w

plane_metric = SegmentationMetric(7)
depth_metric = DepthEstmationMetric(cfg)

image_list_path = glob.glob(f"{cfg.data_path}/image/*")

num_sample = len(image_list_path)
metric = {'pa': [], 'mpa': [], 'miou': [], 'ciou': [], 'rmse': [], 'cor': []}


for image_path in image_list_path:
    image_name = image_path.split('/')[-1][:-4]

    gt_seg_path = os.path.join(cfg.data_path, 'layout_seg', image_name + '.png')
    gt_depth_path = os.path.join(cfg.data_path, 'layout_depth', image_name + '.png')
    gt_keypoint_path = os.path.join(cfg.data_path, 'layout_keypoint', image_name + '.npy')

    gt_seg = io.imread(gt_seg_path)
    gt_seg = cv2.resize(gt_seg, (w, h))
    gt_depth = io.imread(gt_depth_path)/4000
    gt_depth = cv2.resize(gt_depth, (w, h))
    gt_corner = np.load(gt_keypoint_path)

    image = io.imread(image_path)
    image = cv2.resize(image, (w, h))
    image_origin = image.copy()

    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transforms(image)
    image = image.to(device).unsqueeze(0)

    layout_feature, layout_prob, layout_class, pred_surface_normal = network(image)

    with torch.no_grad():
        intrinsic = np.array([[600, 0, 320],
                     [0, 600, 240],
                     [0, 0, 1]])
        intrinsic[0] = intrinsic[0]*w/640
        intrinsic[1] = intrinsic[1]*h/480

        k_inv = np.linalg.inv(intrinsic)

        depth_intersection = DepthIntersection(cfg)

        layout_seg, layout_depth, layout_corner_float, layout_corner_xyz = \
            depth_intersection(image_origin, intrinsic, layout_prob, layout_class, pred_surface_normal)

        pa, cpa, mpa, ciou, miou = plane_metric(layout_class, gt_seg)
        rmse, log10, abs_rel, sq_rel, accu0, accu1, accu2 = depth_metric(layout_depth, gt_depth)
        cor_error = corner_error(h, w, k_inv, layout_corner_xyz, gt_corner)

        metric['pa'].append(pa)
        metric['mpa'].append(mpa)
        metric['miou'].append(miou)
        metric['rmse'].append(rmse)
        metric['cor'].append(cor_error)

final_pa = np.sum(metric['pa'])/num_sample
final_mpa = np.sum(metric['mpa'])/num_sample
final_miou = np.sum(metric['miou'])/num_sample
final_rmse = np.sum(metric['rmse'])/num_sample
final_cor = np.sum(metric['cor'])/num_sample

print('pa :', final_pa)
print('mpa :', final_mpa)
print('miou :', final_miou)
print('rmse :', final_rmse)


