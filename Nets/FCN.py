#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

"""
    创建FCN网络结构
"""


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        # 创建网络结构

        # 先通过最大值池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)

        # C1 kernel=2的卷积层
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU()
        )

        # P1 pooling层
        self.Pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # C2 kernel=2的卷积层
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU()
        )

        # P2 pooling层
        self.Pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # C3 kernel=3的卷积层
        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

        # P3 pooling层
        self.Pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # C4 kernel=3的卷积层
        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

        # P4 pooling层
        self.Pool4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # C5 kernel=3的卷积层
        self.Conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )

        # P5 pooling层
        self.Pool5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # C6 kernel=2的卷积层
        self.Conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        # P6 pooling层
        self.Pool6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # C7 kernel=2的卷积层
        self.Conv7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        # P7 pooling层
        self.Pool7 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # 转置卷积层, 32倍上采样
        self.TransConv32 = nn.ConvTranspose2d(in_channels=1024, out_channels=8, stride=32, kernel_size=32, padding=0,
                                              output_padding=0)

        # 转置卷积层， 16倍上采样
        self.TransConv16_c7 = nn.ConvTranspose2d(in_channels=1024, out_channels=8, stride=2, kernel_size=2, padding=0,
                                                 output_padding=0)
        self.TransConv16_p4 = nn.Conv2d(in_channels=384, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.TransConv16 = nn.ConvTranspose2d(in_channels=8, out_channels=8, stride=16, kernel_size=16, padding=0,
                                              output_padding=0)

        # 转置卷积层，8倍上采样
        self.TransConv8_c7 = nn.ConvTranspose2d(in_channels=1024, out_channels=8, stride=4, kernel_size=4, padding=0,
                                                output_padding=0)
        self.TransConv8_p4 = nn.ConvTranspose2d(in_channels=384, out_channels=8, stride=2, kernel_size=2, padding=0,
                                                output_padding=0)
        self.TransConv8_p3 = nn.Conv2d(in_channels=384, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.TransConv8 = nn.ConvTranspose2d(in_channels=8, out_channels=8, stride=8, kernel_size=8 ,padding=0,
                                             output_padding=0)

    def forward(self, x, option=32):
        """
        :param x: 输入图
        :param option: 上采样的类型
        :return: 输出预测图
        """
        assert option in (8, 16, 32)
        x = self.maxpool(x)

        # 3 torch.Size([384, 128, 128])
        x_p3 = self.Conv3(self.Pool3(self.Conv2(self.Pool2(self.Conv1(self.Pool1(x))))))

        # 4 torch.Size([384, 64, 64])
        x_p4 = self.Conv4(self.Pool4(x_p3))

        # 7 torch.Size([4096, 32, 32])
        x_c7 = self.Conv7(self.Conv6(self.Pool6(self.Conv5(self.Pool5(x_p4)))))

        # 32倍上采样
        if option == 32:
            x_tran32 = self.TransConv32(x_c7)

            return x_tran32

        # 16倍上采样
        elif option == 16:
            x_tran16_c7 = self.TransConv16_c7(x_c7)
            x_tran16_p4 = self.TransConv16_p4(x_p4)
            x_tran16 = self.TransConv16(x_tran16_p4 + x_tran16_c7)

            return x_tran16

        # 8倍上采样
        else:
            x_tran8_c7 = self.TransConv8_c7(x_c7)
            x_tran8_p4 = self.TransConv8_p4(x_p4)
            x_tran8_p3 = self.TransConv8_p3(x_p3)
            x_tran8 = self.TransConv8(x_tran8_p3 + x_tran8_p4 + x_tran8_c7)

            return x_tran8
