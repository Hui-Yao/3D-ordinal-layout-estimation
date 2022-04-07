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
import open3d as o3d

# own
from model.baseline_same import Baseline as UNet
from depth_intersection_module.depth_intersection import DepthIntersection
from depth_intersection_module.misc import  get_layout_pc


parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--image_path', type=str,help='path to image')
parser.add_argument('--pretrained_path', type=str, help='path to pretrained model')
arg = parser.parse_args()


yaml_dir = 'config/layout_config.yaml'
with open(yaml_dir) as f:
    config = yaml.load(f,Loader=yaml.FullLoader)
    cfg = edict(config)

cfg.update(vars(arg))
cfg.visualize = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = UNet(cfg.model)  # resnet101-FPN

resume_dir = cfg.pretrained_path
model_dict = torch.load(resume_dir, map_location=lambda storage, loc: storage)
network.load_state_dict(model_dict)

network.to(device)
network.eval()
h, w = cfg.dataset.h, cfg.dataset.w


image_name = cfg.image_path
image = io.imread(image_name)
image = cv2.resize(image, (cfg.dataset.w, cfg.dataset.h))
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

# visualize
if cfg.visualize == True:
    # 2D visualize
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(image_origin)
    plt.subplot(132)
    plt.imshow(image_origin)
    plt.imshow(layout_seg.reshape(h, w), alpha=0.3)
    plt.subplot(133)
    plt.imshow(layout_depth.reshape(h, w))
    plt.show()

    # 3D visualize
    o3d_pcd_layout = get_layout_pc(image_origin, intrinsic, layout_depth, layout_seg)
    o3d.visualization.draw_geometries([o3d_pcd_layout])

