import torch
import torch.nn as nn
import numpy as np


class SegmentationMetric(nn.Module):
    def __init__(self, numClass):
        super(SegmentationMetric, self).__init__()
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        cIoU = intersection / union
        mIoU = np.nanmean(cIoU)

        return cIoU, mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def forward(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        np.seterr(divide='ignore', invalid='ignore')
        imgPredict = imgPredict.cpu().numpy()
        imgLabel = imgLabel.cpu().numpy()
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

        pa = self.pixelAccuracy()
        cpa = self.classPixelAccuracy()
        mpa = self.meanPixelAccuracy()
        ciou, miou = self.meanIntersectionOverUnion()

        return pa, cpa, mpa, ciou, miou


class DepthEstmationMetric(nn.Module):
    def __init__(self, cfg):
        super(DepthEstmationMetric, self).__init__()
        self.num_pixel = cfg.dataset.h*cfg.dataset.w
        self.num_picture = 0
        self.rmse = 0
        # self.rmse_log = 0
        self.log10 = 0
        self.abs_rel = 0
        self.sq_rel = 0
        self.accu0 = 0
        self.accu1 = 0
        self.accu2 = 0

    def forward(self, pred_depth, gt_depth):
        self.num_picture = self.num_picture + 1

        pred_depth = pred_depth.cpu().reshape(1, -1).numpy()
        gt_depth = gt_depth.cpu().reshape(1, -1).numpy()

        thresh = np.maximum((gt_depth/pred_depth), (pred_depth/gt_depth))
        accu0 = (thresh < 1.25).mean()
        accu1 = (thresh < 1.25 ** 2).mean()
        accu2 = (thresh < 1.25 ** 3).mean()

        rmse = (gt_depth - pred_depth) ** 2
        rmse = np.sqrt(rmse.mean())

        # rmse_log = (np.log(gt_depth) - np.log(pred_depth)) ** 2
        # rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt_depth - pred_depth) / gt_depth)
        sq_rel = np.mean(((gt_depth - pred_depth) ** 2) / gt_depth)

        # err = np.log(pred_depth) - np.log(gt_depth)
        # silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        err = np.abs(np.log10(pred_depth) - np.log10(gt_depth))
        log10 = np.mean(err)

        self.rmse += rmse
        # self.rmse_log += rmse_log
        self.log10 += log10
        self.abs_rel += abs_rel
        self.sq_rel += sq_rel
        self.accu0 += accu0
        self.accu1 += accu1
        self.accu2 += accu2

        _rmse = self.rmse/self.num_picture
        # _rmse_log = self.rmse_log/self.num_picture
        _log10 = self.log10/self.num_picture
        _abs_rel = self.abs_rel/self.num_picture
        _sq_rel = self.sq_rel/self.num_picture
        _accu0 = self.accu0/self.num_picture
        _accu1 = self.accu1/self.num_picture
        _accu2 = self.accu2/self.num_picture

        return  _rmse, _log10, _abs_rel, _sq_rel, _accu0, _accu1, _accu2
        # return  _rmse, _rmse_log, _log10, _abs_rel, _sq_rel, _accu0, _accu1, _accu2







