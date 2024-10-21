#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as f


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


# 分割图像
class Crop_patch(nn.Module):
    def __init__(self, patch_size):
        super(Crop_patch, self).__init__()

        self.patch_size = patch_size

    def forward(self, x):
        """
        :param x: 输入图像，大小为(batch_size, channels, height, weight)
        :return: 分割序列
        """
        feature_shape = (x.shape[-2], x.shape[-1])
        patch_list = []

        assert feature_shape[0] % self.patch_size == 0 & feature_shape[1] % self.patch_size == 0

        num_patches = (feature_shape[0] // self.patch_size, feature_shape[1] // self.patch_size)

        for i in range(num_patches[0]):
            for j in range(num_patches[1]):
                patch = x[:, :, self.patch_size * i:self.patch_size * (i+1),
                          self.patch_size*j:self.patch_size * (j+1)]
                patch_list.append(patch)

        return patch_list, num_patches


def Merge_patches(feature_list, num_patches):
    assert len(feature_list) == num_patches[0] * num_patches[1]

    feature = feature_list[0]
    batch_size, channels, height, width = feature.shape

    for i in range(1, len(feature_list)):
        feature = torch.cat((feature, feature_list[i]), dim=2)

    feature = feature.reshape(batch_size, channels, num_patches[0] * height, num_patches[1] * width)

    return feature


# 基于缩放点积注意力的多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, d_model, heads, **kwargs):
        """
        :param d_model: 多头注意力对应的隐藏层单元数
        :param heads: 多头注意力中头的数量
        :param kwargs: 其他参数
        """
        super(MultiHeadAttention, self).__init__()

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


# 逐位前馈神经网络
class FFN(nn.Module):
    def __init__(self, in_channels, ratios):  # hidden_dim应远大于input_dim
        super(FFN, self).__init__()

        assert isinstance(ratios, int)
        self.Conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*ratios, kernel_size=1, padding=0,
                               stride=1)
        self.Conv2 = nn.Conv2d(in_channels=in_channels*ratios, out_channels=in_channels*ratios, kernel_size=3,
                               padding=1, stride=1)
        self.Conv3 = nn.Conv2d(in_channels=in_channels*ratios, out_channels=in_channels, kernel_size=1, padding=0,
                               stride=1)

    def forward(self, x):
        return f.leaky_relu(self.Conv3(f.leaky_relu(self.Conv2(f.leaky_relu(self.Conv1(x))))))


# HRFormer Block
class HRFormer_Block(nn.Module):
    def __init__(self, in_channels, input_size, patch_size, heads, ratios):
        """
        :param in_channels: 输入通道数
        :param input_size: 输入图像大小
        :param patch_size: 窗口大小
        :param heads: 多头注意力中头的数量
        :param ratios: FFN的扩展比例
        """
        super(HRFormer_Block, self).__init__()

        self.Crop = Crop_patch(patch_size)
        self.Ma = MultiHeadAttention(input_size=patch_size**2, d_model=patch_size, heads=heads)
        self.FFN = FFN(in_channels=in_channels, ratios=ratios)

    def forward(self, x):
        patch_list, num_patches = self.Crop(x)

        for item in patch_list:
            item += self.Ma(item)

        feature = Merge_patches(patch_list, num_patches)

        output = self.FFN(feature)

        return output


# 创建不同的Stage和Transition
class Stage1(nn.Module):
    def __init__(self, modules, blocks):
        super(Stage1, self).__init__()

        self.modules = []

        for i in range(modules):
            module = []
            for j in range(blocks):
                block = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1)
                ).to('cuda:0')
                module.append(block)
            self.modules.append(module)

    def forward(self, x):
        output = 0
        for module in self.modules:
            for block in module:
                output += block(x)

        return output


class Transition1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition1, self).__init__()

        self.get_d4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=1, stride=1,
                                padding=0)
        self.get_d8 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[1], kernel_size=3, stride=2,
                                padding=1)

    def forward(self, x):
        x_d4 = self.get_d4(x)
        x_d8 = self.get_d8(x)

        return x_d4, x_d8


class Stage2(nn.Module):
    def __init__(self, modules, blocks, in_channels, input_size, patch_size, heads, ratios):
        super(Stage2, self).__init__()

        self.modules = modules[1]
        self.blocks = blocks[1]

        self.Module_list_4 = []
        self.Module_list_8 = []

        for i in range(self.modules):
            module_4 = []
            module_8 = []
            for j in range(self.blocks):
                blocks_4 = HRFormer_Block(in_channels=in_channels[0], input_size=input_size//4,
                                          patch_size=patch_size, heads=heads[0], ratios=ratios)
                blocks_8 = HRFormer_Block(in_channels=in_channels[1], input_size=input_size//8,
                                          patch_size=patch_size//2, heads=heads[1], ratios=ratios)

                module_4.append(blocks_4)
                module_8.append(blocks_8)

            self.Module_list_4.append(module_4)
            self.Module_list_8.append(module_8)

    def forward(self, feature_4, feature_8, in_channels):
        for i in range(self.modules):
            for j in range(self.blocks):
                feature_4_block = self.Module_list_4[i][j](feature_4)
                feature_8_block = self.Module_list_8[i][j](feature_8)

                feature_8_d4 = Down_2(in_channels=in_channels[0])(feature_4_block)
                feature_4_d8 = Up_n(in_channels=in_channels[1], n=2)(feature_8_block)

                feature_4 = f.leaky_relu(feature_4_d8 + feature_4_block)
                feature_8 = f.leaky_relu(feature_8_d4 + feature_8_block)

        return feature_4, feature_8


class Transition2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition2, self).__init__()

        self.get_d4 = nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=3, stride=1,
                                padding=1)
        self.get_d8 = nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=3, stride=1,
                                padding=1)
        self.get_d16 = nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[2], kernel_size=3, stride=2,
                                 padding=1)

    def forward(self, x_d4_input, x_d8_input):
        x_d4 = self.get_d4(x_d4_input)
        x_d8 = self.get_d8(x_d8_input)
        x_d16 = self.get_d16(x_d8_input)

        return x_d4, x_d8, x_d16


class Stage3(nn.Module):
    def __init__(self, modules, blocks, in_channels, input_size, patch_size, heads, ratios):
        super(Stage3, self).__init__()

        self.modules = modules[2]
        self.blocks = blocks[2]

        self.Module_list_4 = []
        self.Module_list_8 = []
        self.Module_list_16 = []

        for i in range(self.modules):
            module_4 = []
            module_8 = []
            module_16 = []
            for j in range(self.blocks):
                blocks_4 = HRFormer_Block(in_channels=in_channels[0], input_size=input_size//4,
                                          patch_size=patch_size, heads=heads[0], ratios=ratios)
                blocks_8 = HRFormer_Block(in_channels=in_channels[1], input_size=input_size//8,
                                          patch_size=patch_size//2, heads=heads[1], ratios=ratios)
                blocks_16 = HRFormer_Block(in_channels=in_channels[2], input_size=input_size//16,
                                           patch_size=patch_size//4, heads=heads[2], ratios=ratios)

                module_4.append(blocks_4)
                module_8.append(blocks_8)
                module_16.append(blocks_16)

            self.Module_list_4.append(module_4)
            self.Module_list_8.append(module_8)
            self.Module_list_16.append(module_16)

    def forward(self, feature_4, feature_8, feature_16, in_channels):
        for i in range(self.modules):
            for j in range(self.blocks):
                feature_4_block = self.Module_list_4[i][j](feature_4)
                feature_8_block = self.Module_list_8[i][j](feature_8)
                feature_16_block = self.Module_list_16[i][j](feature_16)

                feature_4_d8 = Up_n(in_channels=in_channels[1], n=2)(feature_8_block)
                feature_4_d16 = Up_n(in_channels=in_channels[2], n=4)(feature_16_block)

                feature_8_d4 = Down_2(in_channels=in_channels[0])(feature_4_block)
                feature_8_d16 = Up_n(in_channels=in_channels[2], n=2)(feature_16_block)

                feature_16_d4 = Down_4(in_channels=in_channels[0])(feature_4_block)
                feature_16_d8 = Down_2(in_channels=in_channels[1])(feature_8_block)

                feature_4 = f.leaky_relu(feature_4_block + feature_4_d8 + feature_4_d16)
                feature_8 = f.leaky_relu(feature_8_block + feature_8_d4 + feature_8_d16)
                feature_16 = f.leaky_relu(feature_16_block + feature_16_d4 + feature_16_d8)

        return feature_4, feature_8, feature_16


class Transition3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition3, self).__init__()

        self.get_d4 = nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=3, stride=1,
                                padding=1)
        self.get_d8 = nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=3, stride=1,
                                padding=1)
        self.get_d16 = nn.Conv2d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=3, stride=1,
                                 padding=1)
        self.get_d32 = nn.Conv2d(in_channels=in_channels[2], out_channels=out_channels[3], kernel_size=3, stride=2,
                                 padding=1)

    def forward(self, x_d4_input, x_d8_input, x_d16_input):
        x_d4 = self.get_d4(x_d4_input)
        x_d8 = self.get_d8(x_d8_input)
        x_d16 = self.get_d16(x_d16_input)
        x_d32 = self.get_d32(x_d16_input)

        return x_d4, x_d8, x_d16, x_d32


class Stage4(nn.Module):
    def __init__(self, modules, blocks, in_channels, input_size, patch_size, heads, ratios):
        super(Stage4, self).__init__()

        self.modules = modules[3]
        self.blocks = blocks[3]

        self.Module_list_4 = []
        self.Module_list_8 = []
        self.Module_list_16 = []
        self.Module_list_32 = []

        for i in range(self.modules):
            module_4 = []
            module_8 = []
            module_16 = []
            module_32 = []
            for j in range(self.blocks):
                blocks_4 = HRFormer_Block(in_channels=in_channels[0], input_size=input_size//4,
                                          patch_size=patch_size, heads=heads[0], ratios=ratios)
                blocks_8 = HRFormer_Block(in_channels=in_channels[1], input_size=input_size//8,
                                          patch_size=patch_size//2, heads=heads[1], ratios=ratios)
                blocks_16 = HRFormer_Block(in_channels=in_channels[2], input_size=input_size//16,
                                           patch_size=patch_size//4, heads=heads[2], ratios=ratios)
                blocks_32 = HRFormer_Block(in_channels=in_channels[3], input_size=input_size//32,
                                           patch_size=patch_size//8, heads=heads[3], ratios=ratios)

                module_4.append(blocks_4)
                module_8.append(blocks_8)
                module_16.append(blocks_16)
                module_32.append(blocks_32)

            self.Module_list_4.append(module_4)
            self.Module_list_8.append(module_8)
            self.Module_list_16.append(module_16)
            self.Module_list_32.append(module_32)

    def forward(self, feature_4, feature_8, feature_16, feature_32, in_channels):
        for i in range(self.modules):
            for j in range(self.blocks):
                feature_4_block = self.Module_list_4[i][j](feature_4)
                feature_8_block = self.Module_list_8[i][j](feature_8)
                feature_16_block = self.Module_list_16[i][j](feature_16)
                feature_32_block = self.Module_list_32[i][j](feature_32)

                feature_4_d8 = Up_n(in_channels=in_channels[1], n=2)(feature_8_block)
                feature_4_d16 = Up_n(in_channels=in_channels[2], n=4)(feature_16_block)
                feature_4_d32 = Up_n(in_channels=in_channels[3], n=8)(feature_32_block)

                feature_8_d4 = Down_2(in_channels=in_channels[0])(feature_4_block)
                feature_8_d16 = Up_n(in_channels=in_channels[2], n=2)(feature_16_block)
                feature_8_d32 = Up_n(in_channels=in_channels[3], n=4)(feature_32_block)

                feature_16_d4 = Down_4(in_channels=in_channels[0])(feature_4_block)
                feature_16_d8 = Down_2(in_channels=in_channels[1])(feature_8_block)
                feature_16_d32 = Up_n(in_channels=in_channels[3], n=2)(feature_32_block)

                feature_32_d4 = Down_8(in_channels=in_channels[0])(feature_4_block)
                feature_32_d8 = Down_4(in_channels=in_channels[1])(feature_8_block)
                feature_32_d16 = Down_2(in_channels=in_channels[2])(feature_16_block)

                feature_4 = f.leaky_relu(feature_4_block + feature_4_d8 + feature_4_d16 + feature_4_d32)
                feature_8 = f.leaky_relu(feature_8_block + feature_8_d4 + feature_8_d16 + feature_8_d32)
                feature_16 = f.leaky_relu(feature_16_block + feature_16_d4 + feature_16_d8 + feature_16_d32)
                feature_32 = f.leaky_relu(feature_32_block + feature_32_d4 + feature_32_d8 + feature_32_d16)

        return feature_4, feature_8, feature_16, feature_32


# 创建HRFormer
class HRFormer(nn.Module):
    def __init__(self, model_type, input_size=256, patch_size=16, ratios=3, classes=8):
        super(HRFormer, self).__init__()

        assert model_type in ('T', 'S', 'B')

        if model_type == 'T':
            self.modules = (1, 1, 3, 2)
            self.blocks = (2, 2, 2, 2)
            self.channels = (18, 36, 72, 144)
            self.heads = (1, 2, 4, 8)

        elif model_type == 'S':
            self.modules = (1, 1, 4, 2)
            self.blocks = (2, 2, 2, 2)
            self.channels = (32, 64, 128, 256)
            self.heads = (1, 2, 4, 8)

        else:
            self.modules = (1, 1, 4, 2)
            self.blocks = (2, 2, 2, 2)
            self.channels = (78, 156, 312, 624)
            self.heads = (2, 4, 8, 16)

        self.input_size = input_size
        self.patch_size = patch_size
        self.ratios = ratios

        self.stage1 = Stage1(modules=self.modules[0], blocks=self.blocks[0])
        self.tran1 = Transition1(256, self.channels)

        self.stage2 = Stage2(modules=self.modules, blocks=self.blocks, in_channels=self.channels, ratios=self.ratios,
                             input_size=self.input_size, patch_size=self.patch_size, heads=self.heads)
        self.tran2 = Transition2(self.channels, self.channels)

        self.stage3 = Stage3(modules=self.modules, blocks=self.blocks, in_channels=self.channels, ratios=self.ratios,
                             input_size=self.input_size, patch_size=self.patch_size, heads=self.heads)
        self.tran3 = Transition3(self.channels, self.channels)

        self.stage4 = Stage4(modules=self.modules, blocks=self.blocks, in_channels=self.channels, ratios=self.ratios,
                             input_size=self.input_size, patch_size=self.patch_size, heads=self.heads)

        self.Up2 = Up_n(in_channels=self.channels[1], n=2)
        self.Up4 = Up_n(in_channels=self.channels[2], n=4)
        self.Up8 = Up_n(in_channels=self.channels[3], n=8)

        self.get_predict = nn.Conv2d(in_channels=self.channels[0], out_channels=classes, kernel_size=1, stride=1)

    def forward(self, x, option=None):
        x_d4 = self.stage1(x)
        x_d4, x_d8 = self.tran1(x_d4)
        x_d4, x_d8 = self.stage2(x_d4, x_d8, self.channels)
        x_d4, x_d8, x_d16 = self.tran2(x_d4, x_d8)
        x_d4, x_d8, x_d16 = self.stage3(x_d4, x_d8, x_d16, self.channels)
        x_d4, x_d8, x_d16, x_d32 = self.tran3(x_d4, x_d8, x_d16)
        x_d4, x_d8, x_d16, x_d32 = self.stage4(x_d4, x_d8, x_d16, x_d32, self.channels)

        x_d4_d8 = self.Up2(x_d8)
        x_d4_d16 = self.Up4(x_d16)
        x_d4_d32 = self.Up8(x_d32)

        output = self.get_predict(f.leaky_relu(x_d4 + x_d4_d8 + x_d4_d16 + x_d4_d32))

        return output
