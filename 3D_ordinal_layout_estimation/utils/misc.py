import torch
from skimage import filters
import cv2
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(parameters, cfg):  # network.parameter(), cfg.solver()
    if cfg.method == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters),  # 过滤出所有参数中需要记录梯度的？
                                    lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.method == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters),
                                     lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.method == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, parameters),
                                        lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.method == 'adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, parameters),
                                         lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def adjust_learning_rate(cfg, optimizer, epoch):
    lr = cfg.solver.lr * (cfg.solver.lr_decay_rate ** (epoch // cfg.solver.lr_decay_interval))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_edge(gt_seg, bandwidth):
    '''
    using the sobel to get the gt_edge from gt_seg
    '''
    gt_seg = np.array(gt_seg.cpu().detach)

    edges =  filters.sobel(gt_seg)
    kernel = np.ones((2, 2), np.uint8)
    gt_edge = cv2.dilate(edges, kernel, iterations=bandwidth)
    mask = gt_edge > 0.0001
    gt_edge[mask] = 1

    # plt.imshow(gt_edge)
    # plt.show()

    return gt_edge

def cal_batch_weight(num_class, gt_seg):
    batch_size, h, w = gt_seg.size()
    num_pixel_batch = batch_size * h * w

    weight = torch.ones(num_class)
    for single_class in range(num_class):
        mask = gt_seg == single_class
        num_pixel_class = torch.sum(mask)
        if num_pixel_class != 0:
            weight_class = num_pixel_batch/(1.*num_class * num_pixel_class)

            weight[single_class] = weight_class

    return weight.cuda()



