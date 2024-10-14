#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as f

"""
    用PyTorch实现HRNet-W32网络结构，网络结构以图为准
"""


# Basic Block
class Basic_Block(nn.Module):
    # 构建Basic Block
    def __init__(self, in_channels):
        super(Basic_Block, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# 上下采样块
class Up_n(nn.Module):
    # 构建n倍上采样块
    def __init__(self, in_channels, n):
        super(Up_n, self).__init__()

        assert in_channels % n == 0

        i_c = in_channels // n
        assert i_c % 2 == 0

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=i_c, kernel_size=1, stride=1, padding=0),

            # BottleNet块
            nn.Conv2d(in_channels=i_c, out_channels=i_c // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=i_c // 2, out_channels=i_c // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=i_c // 2, out_channels=i_c, kernel_size=1, stride=1, padding=0),

            # 转置卷积求解上采样
            nn.ConvTranspose2d(in_channels=i_c, out_channels=i_c, kernel_size=n, stride=n, padding=0, output_padding=0)
        )

    def forward(self, x):
        return self.net(x)


class Down_2(nn.Module):
    # 构建2倍下采样块
    def __init__(self, in_channels):
        super(Down_2, self).__init__()

        assert in_channels % 2 == 0

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=2, padding=1),

            # BottleNeck块
            nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels // 2, out_channels=2 * in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x)


class Down_4(nn.Module):
    # 构建4倍下采样块
    def __init__(self, in_channels):
        super(Down_4, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=4*in_channels, kernel_size=3, stride=2, padding=1),

            # BottleNeck块
            nn.Conv2d(in_channels=4*in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=4*in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x)


class Down_8(nn.Module):
    # 构建8倍下采样块
    def __init__(self, in_channels):
        super(Down_8, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=8*in_channels, kernel_size=3, stride=2, padding=1),

            # BottleNeck块
            nn.Conv2d(in_channels=8*in_channels, out_channels=2*in_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=2*in_channels, out_channels=2*in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=2*in_channels, out_channels=8*in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x)


# Transition
class Transition1(nn.Module):
    def __init__(self):
        super(Transition1, self).__init__()
        self.Conv_s1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.Conv_s2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x_c32 = self.Conv_s1(x)
        x_c64 = self.Conv_s2(x)

        return x_c32, x_c64


class Transition2(nn.Module):
    def __init__(self):
        super(Transition2, self).__init__()
        self.Conv_s2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x_c128 = self.Conv_s2(x)

        return x_c128


class Transition3(nn.Module):
    def __init__(self):
        super(Transition3, self).__init__()
        self.Conv_s2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x_c256 = self.Conv_s2(x)

        return x_c256


# 创建3个stage结构
class Stage2(nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()

        # 下采样块
        self.Down2 = Down_2(in_channels=32)

        # 上采样块
        self.Up2 = Up_n(in_channels=64, n=2)

        # 两个Basic Block
        self.Basic_Block_32 = Basic_Block(in_channels=32)
        self.Basic_Block_64 = Basic_Block(in_channels=64)

    def forward(self, x_c32, x_c64):
        x_down2 = self.Down2(self.Basic_Block_32(x_c32))
        x_up2 = self.Up2(self.Basic_Block_64(x_c64))

        x_c32 = f.relu(x_c32 + x_up2)
        x_c64 = f.relu(x_c64 + x_down2)

        return x_c32, x_c64


class Stage3(nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()

        # 三个Basic Block
        self.Basic_Block_32 = Basic_Block(in_channels=32)
        self.Basic_Block_64 = Basic_Block(in_channels=64)
        self.Basic_Block_128 = Basic_Block(in_channels=128)

        # 通道32的特征图需要一个2倍下采样块，一个4倍下采样块
        self.Down2_32 = Down_2(in_channels=32)
        self.Down4_32 = Down_4(in_channels=32)

        # 通道64的特征图需要一个2倍上采样块，一个2倍下采样块
        self.Down2_64 = Down_2(in_channels=64)
        self.Up2_64 = Up_n(in_channels=64, n=2)

        # 通道128的特征图需要一个2倍上采样块，一个4倍上采样块
        self.Up2_128 = Up_n(in_channels=128, n=2)
        self.Up4_128 = Up_n(in_channels=128, n=4)

    def forward(self, x_c32, x_c64, x_c128):
        """
        :param x_c32: 通道数为32的特征图
        :param x_c64: 通道数为64的特征图
        :param x_c128: 通道数为128的特征图
        :return: x_c32, x_c64, x_c128
        """
        # 先通过Basic Block块
        x_c32 = self.Basic_Block_32(x_c32)
        x_c64 = self.Basic_Block_64(x_c64)
        x_c128 = self.Basic_Block_128(x_c128)

        # 对通道数为32的特征图下采样
        x_c32_d2 = self.Down2_32(x_c32)
        x_c32_d4 = self.Down4_32(x_c32)

        # 对通道数为64的特征图进行上采样和下采样
        x_c64_u2 = self.Up2_64(x_c64)
        x_c64_d2 = self.Down2_64(x_c64)

        # 对通道数为128的特征图进行上采样
        x_c128_u2 = self.Up2_128(x_c128)
        x_c128_u4 = self.Up4_128(x_c128)

        # 获取输出
        x_c32 = f.relu(x_c32 + x_c64_u2 + x_c128_u4)
        x_c64 = f.relu(x_c64 + x_c32_d2 + x_c128_u2)
        x_c128 = f.relu(x_c128 + x_c32_d4 + x_c64_d2)

        return x_c32, x_c64, x_c128


class Stage4(nn.Module):
    def __init__(self):
        super(Stage4, self).__init__()

        # 四个Basic Block
        self.Basic_Block_32 = Basic_Block(in_channels=32)
        self.Basic_Block_64 = Basic_Block(in_channels=64)
        self.Basic_Block_128 = Basic_Block(in_channels=128)
        self.Basic_Block_256 = Basic_Block(in_channels=256)

        # 对通道数为32的特征图，进行2,4,8倍下采样
        self.Down2_32 = Down_2(in_channels=32)
        self.Down4_32 = Down_4(in_channels=32)
        self.Down8_32 = Down_8(in_channels=32)

        # 对通道为64的特征图，进行2倍上采样，2，4倍下采样
        self.Up2_64 = Up_n(in_channels=64, n=2)
        self.Down2_64 = Down_2(in_channels=64)
        self.Down4_64 = Down_4(in_channels=64)

        # 对通道为128的特征图，进行2，4倍上采样，2倍下采样
        self.Up2_128 = Up_n(in_channels=128, n=2)
        self.Up4_128 = Up_n(in_channels=128, n=4)
        self.Down2_128 = Down_2(in_channels=128)

        # 对通道为256的特征图，进行2，4，8倍上采样
        self.Up2_256 = Up_n(in_channels=256, n=2)
        self.Up4_256 = Up_n(in_channels=256, n=4)
        self.Up8_256 = Up_n(in_channels=256, n=8)

    def forward(self, x_c32, x_c64, x_c128, x_c256):
        # 先通过Basic Block块
        x_c32 = self.Basic_Block_32(x_c32)
        x_c64 = self.Basic_Block_64(x_c64)
        x_c128 = self.Basic_Block_128(x_c128)
        x_c256 = self.Basic_Block_256(x_c256)

        # 对通道数为32的特征图进行处理
        x_c32_d2 = self.Down2_32(x_c32)
        x_c32_d4 = self.Down4_32(x_c32)
        x_c32_d8 = self.Down8_32(x_c32)

        # 对通道数为64的特征图进行处理
        x_c64_u2 = self.Up2_64(x_c64)
        x_c64_d2 = self.Down2_64(x_c64)
        x_c64_d4 = self.Down4_64(x_c64)

        # 对通道数为128的特征图进行处理
        x_c128_u2 = self.Up2_128(x_c128)
        x_c128_u4 = self.Up4_128(x_c128)
        x_c128_d2 = self.Down2_128(x_c128)

        # 对通道数为256的特征图进行处理
        x_c256_u2 = self.Up2_256(x_c256)
        x_c256_u4 = self.Up4_256(x_c256)
        x_c256_u8 = self.Up8_256(x_c256)

        # 获取输出
        x_c32 = f.relu(x_c32 + x_c64_u2 + x_c128_u4 + x_c256_u8)
        x_c64 = f.relu(x_c32_d2 + x_c64 + x_c128_u2 + x_c256_u4)
        x_c128 = f.relu(x_c32_d4 + x_c64_d2 + x_c128 + x_c256_u2)
        x_c256 = f.relu(x_c32_d8 + x_c64_d4 + x_c128_d2 + x_c256)

        return x_c32, x_c64, x_c128, x_c256


# Stage4的后处理模块
class Sub_produce(nn.Module):
    def __init__(self):
        super(Sub_produce, self).__init__()
        # 最后四个Basic Block块以及上采样块
        self.Basic_Block_32 = Basic_Block(in_channels=32)
        self.Basic_Block_64 = Basic_Block(in_channels=64)
        self.Basic_Block_128 = Basic_Block(in_channels=128)
        self.Basic_Block_256 = Basic_Block(in_channels=256)

        self.upsample_64 = Up_n(in_channels=64, n=2)
        self.upsample_128 = Up_n(in_channels=128, n=4)
        self.upsample_256 = Up_n(in_channels=256, n=8)

    def forward(self, x_c32, x_c64, x_c128, x_c256):
        x_c32 = self.Basic_Block_32(x_c32)
        x_c64 = self.Basic_Block_64(x_c64)
        x_c128 = self.Basic_Block_128(x_c128)
        x_c256 = self.Basic_Block_256(x_c256)

        output = f.relu(x_c32 + self.upsample_64(x_c64) + self.upsample_128(x_c128) + self.upsample_256(x_c256))

        return output


# HRNet
class HRNet(nn.Module):
    def __init__(self):
        super(HRNet, self).__init__()

        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=4, padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
        )

        # Transition1块
        self.tran1 = Transition1()

        # Stage2块
        self.stage2 = Stage2()

        # Transition2块
        self.tran2 = Transition2()

        # Stage3块
        self.Stage3 = nn.Sequential(
            Stage3(),
            Stage3(),
            Stage3(),
            Stage3(),
        )

        # Transition3块
        self.tran3 = Transition3()

        # Stage4块
        self.Stage4 = nn.Sequential(
            Stage4(),
            Stage4(),
        )

        # 最后处理
        self.Sub_pro = Sub_produce()

        self.upsample = Up_n(in_channels=32, n=4)

    def forward(self, x, option=None):
        x_ds = self.downsample(x)

        x_c32, x_c64 = self.tran1(x_ds)

        x_c32, x_c64_stage2 = self.stage2(x_c32, x_c64)

        x_c128 = self.tran2(x_c64)

        for layer in self.Stage3:
            x_c32, x_c64, x_c128 = layer(x_c32, x_c64, x_c128)

        x_c256 = self.tran3(x_c128)

        for layer in self.Stage4:
            x_c32, x_c64, x_c128, x_c256 = layer(x_c32, x_c64, x_c128, x_c256)

        output = self.upsample(self.Sub_pro(x_c32, x_c64, x_c128, x_c256))

        return output
