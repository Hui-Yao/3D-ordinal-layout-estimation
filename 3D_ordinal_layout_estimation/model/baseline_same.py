import torch
import torch.nn as nn
import model.resnet_scene as resnet


class ResNet(nn.Module):
    def __init__(self, orig_resnet):
        super(ResNet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1

        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2

        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3

        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x1 = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x1)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x1, x2, x3, x4, x5


class Baseline(nn.Module):
    def __init__(self, cfg):    # 这里cfg指的是cfg.model：   arch: resnet101  pretrained: True  embed_dims: 2   fix_bn: False
        super(Baseline, self).__init__()

        orig_resnet = resnet.__dict__[cfg.arch](pretrained=cfg.pretrained)  # resnet.__dict__[cfg.arch]表示对应的50或者101函数，后面的括号表示调用该函数
        self.backbone = ResNet(orig_resnet)     #
        '''
        orig_resnet是一个resnet_scene.py/Resnet类实例，然后在本文件中复制了一个orig_resnet, em~~~~~
        也就得到了encoder
        '''
        self.relu = nn.ReLU(inplace=True)

        channel = 64
        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv2 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv1 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv0 = nn.Conv2d(channel, channel, (1, 1))

        # lateral 侧面的，横向的
        self.c5_conv = nn.Conv2d(2048, channel, (1, 1))
        self.c4_conv = nn.Conv2d(1024, channel, (1, 1))
        self.c3_conv = nn.Conv2d(512, channel, (1, 1))
        self.c2_conv = nn.Conv2d(256, channel, (1, 1))
        self.c1_conv = nn.Conv2d(128, channel, (1, 1))

        self.p0_conv = nn.Conv2d(channel, channel, (3, 3), padding=1)

        # pred num_wall semantic mask   假设最多只有8个布局平面
        self.pred_layout = nn.Conv2d(channel, 7, (1, 1), padding=0)
        # pred plane parameters of abcd
        self.pred_normal = nn.Conv2d(channel, 3, (1, 1), padding=0)

    def top_down(self, x):
        c1, c2, c3, c4, c5 = x

        p5 = self.relu(self.c5_conv(c5))
        p4 = self.up_conv5(self.upsample(p5)) + self.relu(self.c4_conv(c4))
        p3 = self.up_conv4(self.upsample(p4)) + self.relu(self.c3_conv(c3))
        p2 = self.up_conv3(self.upsample(p3)) + self.relu(self.c2_conv(c2))
        p1 = self.up_conv2(self.upsample(p2)) + self.relu(self.c1_conv(c1))

        p0 = self.upsample(p1)

        p0 = self.relu(self.p0_conv(p0))

        return p0, p1, p2, p3, p4, p5

    def forward(self, x):
        # bottom up
        c1, c2, c3, c4, c5 = self.backbone(x)  # 依次是conv3， layer1， layer2, layer3, layer4的输出

        # top down
        p0, p1, p2, p3, p4, p5 = self.top_down((c1, c2, c3, c4, c5))    # p0——1/2(h, w)

        # output
        layout_feature_8 = self.pred_layout(p0)  # (batch_size, 8, h, w)
        layout_prob_8 = torch.nn.functional.softmax(layout_feature_8, dim=1)

        layout_prob_1 = torch.max(layout_prob_8, 1)[0]      # 0代表返回8通道方向上最大的概率值
        layout_class_1 = torch.max(layout_prob_8, 1)[1]     # 1代表返回8通道方向上最大概率所在的通道序号

        layout_feature_1 = torch.max(layout_feature_8, 1)[0]  # (batch_size, h, w), 这个像素属于这个类别的概率
        # layout_class_1 = torch.max(layout_prob_8, 1)[1]  # (batch_size, h, w)， 这个类别在第几个通道

        surface_normal = self.pred_normal(p0)

        return layout_feature_8, layout_prob_8, layout_class_1, surface_normal
        # return layout_prob_8, layout_prob_1, layout_class_1
