"""
Copy from https://github.com/CSAILVision/semantic-segmentation-pytorch
"""

import os
import torch
import torch.nn as nn
import math

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


__all__ = ['ResNet', 'resnet50', 'resnet101']


model_urls = {
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    '''
    一个Bottleneck 残差模块中包含3个3X3的卷积，3个relu，3个BN
    如果F(x)相对于x，FM的大小和通道数都没有变化，那么就不用对x做处理；否则要先对x做处理才能相加

    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # stride是默认为1啊
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,    # 第二个卷积使得FM的长宽各缩小一半， downsaple使得分支也缩小一半
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)  # 通道融合， 通道数翻4倍
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None: #
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 默认的Conv3x3, kernel_size = 3, padding = 1, stride = 1，所以不对这三个参数做改变的话，featuremap长宽不变。
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):  # block=Bottleneck, layers=[3, 4, 23, 3],
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)  # 输入3通道的RGB图像，输出64通道的feature map， 长宽变为1/2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)    # 通道数64，长宽不变
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)   # 通道数变为128，长宽不变
        self.bn3 = nn.BatchNorm2d(128)  # encoder输出的FM的channel为128
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    # FM长宽变为[n+1/2],非整数向左取整，
        # maxpool输出FM长宽为64，通道数为128，使得FM长宽变为1/4

        self.layer1 = self._make_layer(block, 64, layers[0])                # 输出通道数为64*4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)     # 输出通道数为128*4，FM长宽变为1/8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)     # 输出通道数为256*4，FM长宽变为1/16
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)     # 输出通道数为512*4，FM长宽变为1/32   如果输入为192*256，到这里6*8
        self.avgpool = nn.AvgPool2d(7, stride=1)                            # 平均池化，kernel_size = 7
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels    # 计算每个卷积操作的参数量
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        #  block=Bottleneck, planes=64 , blocks=3        self.inplanes = 128    输出64*4  (planes*4)
        #  block=Bottleneck, planes=128, blocks=4        self.inplanes = 64*4   输出128*4 (planes*4)
        #  block=Bottleneck, planes=256, blocks=23       self.inplanes = 128*4  输出256*4 (planes*4)
        #  block=Bottleneck, planes=512, blocks=3        self.inplanes = 256*4  输出512*4 (planes*4)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 相对于输入，如果输出1的FM的H&W变化了或者通道数改变了，
                                                                      # 那么就要对输入进行1X1‘下采样’，使得下采样后的输出2和输出1的H&W及channel相同
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))     # 加了一个block
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):  # 加了[3, 4, 23, 3]个block
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)   # 输入x.shape = (512*4,12,16)，所以x[0] = 2048
        x = self.fc(x)  #

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=True)
    return model


def resnet101(pretrained=False, **kwargs):  # pretrained = True
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet101']), strict=True)
    return model


def load_url(url):
    torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
    model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    return torch.load(cached_file, map_location=lambda storage, loc: storage)


if __name__ == '__main__':
    model = resnet101(pretrained=True)
