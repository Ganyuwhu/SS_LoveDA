#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as f


# 下采样阶段所使用的模块
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.Conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.Downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input_feature):
        return self.Downsample(f.relu(self.Conv2(f.relu(self.Conv1(input_feature)))))


# 上采样阶段所使用的模块
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()

        assert isinstance(in_channels, tuple)  # 第一个元素为同层的输入通道数，第二个元素为下一层到的输入通道数

        self.in_channels = in_channels
        self.middle_channels = 2*out_channels
        self.out_channels = out_channels

        self.UpSample = nn.ConvTranspose2d(in_channels=self.in_channels[1], out_channels=self.middle_channels,
                                           stride=2, kernel_size=2, padding=0, output_padding=0)

        self.Conv = nn.Conv2d(in_channels=self.in_channels[0]+self.middle_channels, out_channels=self.out_channels,
                              stride=1, kernel_size=3, padding=1)

    def forward(self, input1, input2):
        """
        :param input1: 低级特征，维度更高，通道更少
        :param input2: 高级特征，维度较低，通道更多，需要减少通道数并提高维度
        :return: 输出图
        """

        return f.relu(self.Conv(torch.cat((input1, self.UpSample(input2)), dim=1)))


# UNet网络架构
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Pooling层降低维度
        self.Pool = nn.AvgPool2d(kernel_size=3, stride=4, padding=1)

        # 1、池化层提升通道数
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        # 2、downsample阶段
        self.DownSample1 = DownsampleBlock(in_channels=32, out_channels=64)
        self.DownSample2 = DownsampleBlock(in_channels=64, out_channels=128)
        self.DownSample3 = DownsampleBlock(in_channels=128, out_channels=256)
        self.DownSample4 = DownsampleBlock(in_channels=256, out_channels=256)

        # 3、upsample阶段
        self.UpSample4 = UpsampleBlock(in_channels=(256, 256), out_channels=128)
        self.UpSample3 = UpsampleBlock(in_channels=(128, 128), out_channels=64)
        self.UpSample2 = UpsampleBlock(in_channels=(64, 64), out_channels=32)
        self.UpSample1 = UpsampleBlock(in_channels=(32, 32), out_channels=16)

        # 4、获取预测图
        self.get_predict = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)

    def forward(self, x, option=None):
        x = self.Pool(x)

        x_c1 = self.Conv1(x)

        x_d1 = self.DownSample1(x_c1)

        x_d2 = self.DownSample2(x_d1)

        x_d3 = self.DownSample3(x_d2)

        x_d4 = self.DownSample4(x_d3)

        x_u4 = self.UpSample4(x_d3, x_d4)

        x_u3 = self.UpSample3(x_d2, x_u4)

        x_u2 = self.UpSample2(x_d1, x_u3)

        x_u1 = self.UpSample1(x_c1, x_u2)

        predict_feature = self.get_predict(x_u1)

        return predict_feature
