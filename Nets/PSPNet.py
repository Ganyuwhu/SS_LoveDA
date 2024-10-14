#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as f

"""
    构建PSPNet网络结构
"""


class ResBlock(nn.Module):
    """
        构建残差网络块
    """
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0, stride=1)
        self.Conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1)
        self.Conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=2)
        self.weight = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        x_out = f.relu(self.Conv3(f.relu(self.Conv2(f.relu(self.Conv1(x))))))

        x_res = f.relu(self.weight(x))

        return x_out + x_res


class ResNet_down8(nn.Module):
    """
        8倍下采样的ResNet
    """
    def __init__(self):
        super(ResNet_down8, self).__init__()
        # 先经过一个7*7的卷积层，进行2倍下采样
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2)

        # ResBlock1，进行2倍下采样
        self.Res1 = ResBlock(in_channels=64, out_channels=128)

        # ResBlock2,进行2倍下采样
        self.Res2 = ResBlock(in_channels=128, out_channels=256)

    def forward(self, x):
        x = self.Res2(self.Res1(f.relu(self.Conv1(x))))
        return x


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
        self.resnet = ResNet_down8()
        self.ppm = PyramidPoolingModule(in_channels=256, bin_size_list=[1, 2, 3, 6])
        self.Conv = nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, stride=1, padding=1)

    def forward(self, x, option=None):
        x_res = self.resnet(x)
        x_ppm = self.ppm(x_res)
        x_predict = torch.cat((x_ppm, x_res), dim=1)
        x_final = self.Conv(f.interpolate(x_predict, size=(1024, 1024), mode='bilinear', align_corners=True))

        return x_final
