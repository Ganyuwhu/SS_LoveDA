#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

"""
    从头构建MANet
"""


# ResNeXt，不下采样
class ResNeXt_Block(nn.Module):
    def __init__(self, in_channels, base_width, groups=32):
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

    def forward(self, x):
        x_conv1 = f.relu(self.bn1(self.Conv1(x)))
        x_conv2 = f.relu(self.bn2(self.Conv2(x_conv1)))
        x_conv3 = f.relu(self.bn3(self.Conv3(x_conv2)))
        x_conv4 = f.relu(self.bn4(self.Conv4(x_conv3)))

        output = x + x_conv4

        return output


# 2倍下采样块
class DownSample_2(nn.Module):
    def __init__(self, in_channels):
        super(DownSample_2, self).__init__()

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


# Channel Attention块
class Channel_Attention(nn.Module):
    def __init__(self):
        super(Channel_Attention, self).__init__()

    def forward(self, x):
        """
        :param x: 输入图，大小为(batch_size, channels, height, width)
        :return: 输出图
        """
        shape = x.shape
        reshape_x = x.reshape(shape[0], shape[1], -1)
        transpose_x = reshape_x.reshape(shape[0], -1, shape[1])

        weight = torch.einsum('bcs, bsd -> bcd', [reshape_x, transpose_x])

        weight_value = torch.einsum('bcd, bds -> bcs', [weight, reshape_x])

        output = (reshape_x + weight_value).reshape(shape)

        return output


# Kernel Attention块
class Kernel_Attention(nn.Module):
    def __init__(self, input_size, d_model):
        super(Kernel_Attention, self).__init__()

        self.input_size = input_size

        self.get_key = nn.Linear(input_size, d_model)
        self.get_query = nn.Linear(input_size, d_model)
        self.get_value = nn.Linear(input_size, d_model)

        self.fc_out = nn.Linear(d_model, input_size)

    def forward(self, x):
        reshape_x = x.reshape(x.shape[0], x.shape[1], -1)
        assert reshape_x.shape[2] == self.input_size

        key = self.get_key(reshape_x)
        query = self.get_query(reshape_x)
        value = self.get_value(reshape_x)

        shape = key.shape  # batch_size, channels, d_model

        softplus_key = f.softplus(key).permute(0, 2, 1)
        softplus_query = f.softplus(query)

        weight = torch.einsum('bcd, bdh -> bch', [softplus_query, softplus_key])

        result = torch.einsum('bch, bhd -> bcd', [weight, value])

        output = self.fc_out(result).reshape(shape[0], shape[1], x.shape[2], x.shape[3])

        return output


# Attention块
class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, input_size):
        super(Attention, self).__init__()
        assert isinstance(input_size, int)

        self.Conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        # 路径1 CAM
        self.cam_bn1 = nn.BatchNorm2d(in_channels)
        self.cam = Channel_Attention()
        self.cam_bn2 = nn.BatchNorm2d(out_channels)

        # 路径2 KAM
        self.kam_bn1 = nn.BatchNorm2d(in_channels)
        self.kam = Kernel_Attention(input_size, d_model=int(np.sqrt(input_size)))
        self.kam_bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        cam_x = f.relu(self.cam_bn2(self.Conv2(self.cam(f.relu(self.cam_bn1(self.Conv1(x)))))))
        kam_x = f.relu(self.kam_bn2(self.Conv2(self.kam(f.relu(self.kam_bn1(self.Conv1(x)))))))

        return cam_x + kam_x


# 反卷积层
class Deconvolution(nn.Module):
    def __init__(self, feature_shape):
        super(Deconvolution, self).__init__()

        assert isinstance(feature_shape, tuple)

        self.size = (2*feature_shape[0], 2*feature_shape[1])

    def forward(self, x):
        return f.interpolate(x, size=self.size, mode='bilinear', align_corners=True)


# MANet
class MANet(nn.Module):
    def __init__(self, feature_shape):
        super(MANet, self).__init__()

        assert feature_shape == (256, 256)

        # 变形
        self.Reshape = nn.AvgPool2d(kernel_size=3, stride=4, padding=1)

        # 图中的第一行模块
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.Res2 = ResNeXt_Block(in_channels=64, base_width=[64, 64, 128], groups=32)
        self.Res3 = ResNeXt_Block(in_channels=128, base_width=[128, 128, 256], groups=32)
        self.Res4 = ResNeXt_Block(in_channels=256, base_width=[256, 256, 512], groups=32)
        self.Res5 = ResNeXt_Block(in_channels=512, base_width=[512, 512, 1024], groups=32)

        # 第二行，DownSample块
        self.DownSample1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.DownSample2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.DownSample3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.DownSample4 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.DownSample5 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # 第三行，Attention块
        self.Attention1 = Attention(in_channels=768, out_channels=256, input_size=16*16)
        self.Attention2 = Attention(in_channels=512, out_channels=128, input_size=32*32)
        self.Attention3 = Attention(in_channels=256, out_channels=64, input_size=64*64)
        self.Attention4 = Attention(in_channels=64, out_channels=8, input_size=128*128)

        # 第四行，反卷积块
        self.Deconv1 = Deconvolution(feature_shape=(128, 128))
        self.Deconv2 = Deconvolution(feature_shape=(64, 64))
        self.Deconv3 = Deconvolution(feature_shape=(32, 32))
        self.Deconv4 = Deconvolution(feature_shape=(16, 16))
        self.Deconv5 = Deconvolution(feature_shape=(8, 8))

        # 跟在ResBlock后的下采样层
        self.AfterConv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.AfterRes2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.AfterRes3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.AfterRes4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)

    def forward(self, x, option=None):
        x = self.Reshape(x)

        # 获取第一行的输出
        x_conv1 = self.Conv1(x)
        x_res2 = self.Res2(self.AfterConv1(x_conv1))
        x_res3 = self.Res3(self.AfterRes2(x_res2))
        x_res4 = self.Res4(self.AfterRes3(x_res3))
        x_res5 = self.Res5(self.AfterRes4(x_res4))

        # 第一行输出做DownSample
        x_ds_conv1 = self.DownSample1(x_conv1)
        x_ds_res2 = self.DownSample2(x_res2)
        x_ds_res3 = self.DownSample3(x_res3)
        x_ds_res4 = self.DownSample4(x_res4)
        x_ds_res5 = self.DownSample5(x_res5)

        # 进行注意力和反卷积操作
        x_dec5 = self.Deconv5(x_ds_res5)
        x_dec4 = self.Deconv4(self.Attention1(torch.cat((x_dec5, x_ds_res4), dim=1)))
        x_dec3 = self.Deconv3(self.Attention2(torch.cat((x_dec4, x_ds_res3), dim=1)))
        x_dec2 = self.Deconv2(self.Attention3(torch.cat((x_dec3, x_ds_res2), dim=1)))
        x_dec1 = self.Deconv1(self.Attention4(torch.cat((x_dec2, x_ds_conv1), dim=1)))

        # 获取输出
        output = f.interpolate(x_dec1, size=(256, 256), mode='bilinear', align_corners=True)

        return output

