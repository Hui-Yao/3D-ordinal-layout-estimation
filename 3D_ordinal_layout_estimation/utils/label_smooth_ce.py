import torch
import torch.nn as nn

class NMTCritierion(nn.Module):
    """
    TODO:
    1. Add label smoothing
    """

    def __init__(self, label_smoothing=0.0, weight=0, num_tokens = 8):      # label_smoothing: 偏移的值
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.weight = weight
        self.num_tokens = num_tokens

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='none')   # 用smoothing就用KL散度， kl散度可以是小数
            # self.criterion = nn.KLDivLoss(size_average=True)   # 用smoothing就用KL散度， kl散度可以是小数
        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=100000)    # NLLLoss只能用整数，所以无法用在label smoothing中
        self.confidence = 1.0 - label_smoothing     # 降低后的原类别的标签

    def _smooth_label(self, num_tokens):
        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.

        # one_hot = torch.randn(1, num_tokens)    # 创建一个[1, n]的二维数组，用torch.zeros不也可以么
        one_hot = torch.zeros((1, num_tokens))
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))  # 对于一个样本(像素)，创建一个（1， num_tokens）的向量
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels, face):
        # print(-99, dec_outs.shape)
        # print(self.weight.device)
        # print(face.device)
        # print(-112, self.weight.shape)
        weight = self.weight * face
        # print(-111, weight.shape)

        dec_outs = dec_outs.view(self.num_tokens, -1).T     # (hw, class)
        scores = self.LogSoftmax(dec_outs)

        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(self.num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M] ,复制hw个，即为每一个像素创建一个one-hot，其中填充的数值均为α/k-1
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1], 将GT位置的数值替换成confidence
            gtruth = tmp_.detach()  # (hw, class)


        loss_ = self.criterion(scores, gtruth).mean(dim=0)  # 求得8个维度上每个维度的平均损失
        # print(100, loss_.shape)
        # print(101, loss_.dtype)
        # print(102, weight.dtype)

        # similar to cross-entropy, just cal the loss in the channel of GT
        loss= torch.sum(loss_ * weight)/sum(weight)     # 对每个维度的损失加权，并且除以权重的和
        # print(111, scores.shape)
        # print(222, gtruth.shape)
        # print(333, loss.shape)
        return loss