#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as f

"""
    构建PSPNet网络结构
"""


# ResNeXt，不下采样
class ResNeXt_Block(nn.Module):
    def __init__(self, in_channels, out_channels, base_width, groups=32):
        """
        :param in_channels: 输入通道数
        :param base_width: 列表，用来表示每条路径上的通道数，列表的索引表示对应次序的块
        :param groups: 路径数，默认为32
        """
        super(ResNeXt_Block, self).__init__()

        assert len(base_width) == 3  # ResNeXt的中间Block有三个

        # 获取中间块的通道数
        self.middle_channels = base_width

        # 第一层，分组卷积
        self.Conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.middle_channels[0], kernel_size=1, stride=1,
                               padding=0, groups=groups)

        self.bn1 = nn.BatchNorm2d(self.middle_channels[0])

        # 第二层，分组卷积
        self.Conv2 = nn.Conv2d(in_channels=self.middle_channels[0], out_channels=self.middle_channels[1],
                               kernel_size=3, stride=1, padding=1, groups=groups)

        self.bn2 = nn.BatchNorm2d(self.middle_channels[1])

        # 第三层，分组卷积
        self.Conv3 = nn.Conv2d(in_channels=self.middle_channels[1], out_channels=self.middle_channels[2],
                               kernel_size=1, stride=1, padding=0, groups=groups)

        self.bn3 = nn.BatchNorm2d(self.middle_channels[2])

        # 第四层，卷积改变通道数
        self.Conv4 = nn.Conv2d(in_channels=self.middle_channels[2], out_channels=in_channels,
                               kernel_size=3, stride=1, padding=1)

        self.bn4 = nn.BatchNorm2d(in_channels)

        # 第五层，降维
        self.DownSample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                    padding=1)

    def forward(self, x):
        x_conv1 = f.relu(self.bn1(self.Conv1(x)))
        x_conv2 = f.relu(self.bn2(self.Conv2(x_conv1)))
        x_conv3 = f.relu(self.bn3(self.Conv3(x_conv2)))
        x_conv4 = f.relu(self.bn4(self.Conv4(x_conv3)))

        output = self.DownSample(x + x_conv4)

        return output


class PyramidPoolingModule(nn.Module):
    """
        金字塔池化模块
    """
    def __init__(self, in_channels, bin_size_list):
        super(PyramidPoolingModule, self).__init__()

        self.num_filters = in_channels // len(bin_size_list)
        self.features = nn.ModuleList()

        for i in range(len(bin_size_list)):
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size_list[i]),
                    nn.Conv2d(in_channels=in_channels, out_channels=self.num_filters, kernel_size=1),
                    nn.ReLU()
                )
            )

    # 前向传播
    def forward(self, x):
        outs = []
        for idx, block in enumerate(self.features):
            block_out = block(x)
            block_out = f.interpolate(input=block_out, size=x.shape[2:], mode='bilinear', align_corners=True)
            outs.append(block_out)

        output = torch.cat(outs, dim=1)

        return output


class PSPNet(nn.Module):
    def __init__(self):
        super(PSPNet, self).__init__()
        """
            构建PSPNet，它有三个组成部分，首先是利用ResNet获取特征图，然后再用ppm获取连接图，最后用卷积层生成预测图
        """
        self.Avgpool = nn.AvgPool2d(kernel_size=3, stride=4, padding=1)
        self.DownSample = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.resnet_1 = ResNeXt_Block(in_channels=32, out_channels=64, base_width=[32, 32, 64], groups=32)
        self.resnet_2 = ResNeXt_Block(in_channels=64, out_channels=128, base_width=[64, 64, 128], groups=32)
        self.ppm = PyramidPoolingModule(in_channels=128, bin_size_list=[1, 2, 3, 6])
        self.Conv = nn.Conv2d(in_channels=256, out_channels=8, kernel_size=3, stride=1, padding=1)

    def forward(self, x, option=None):
        x_DownSample = self.DownSample(x)

        x_res1 = self.resnet_1(x_DownSample)

        x_res2 = self.resnet_2(x_res1)

        x_ppm = self.ppm(x_res2)

        x_predict = torch.cat((x_ppm, x_res2), dim=1)

        x_final = self.Conv(f.interpolate(x_predict, size=(256, 256), mode='bilinear', align_corners=True))

        return x_final
