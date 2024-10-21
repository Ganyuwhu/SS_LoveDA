#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    加载数据集
"""

import os
import numpy as np
import random
import shutil
import torch
import torch.nn as nn
import torchvision.transforms

import torchvision.transforms.functional as tf
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from Nets import Precision


# 获取transform
def get_totensor():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return transform


def get_topil():
    transform = transforms.Compose([
        transforms.ToPILImage()
    ])

    return transform


# 定义数据集类
class LoveDA_Dataset(Dataset):
    # 初始化数据集
    def __init__(self, image_root_dir, mask_root_dir, transform=None):
        """
        :param image_root_dir: 图像数据集根目录，由于LoveDA数据集分为urban和rural两部分，因此用容量为2的列表表示目录
        :param mask_root_dir: 掩码数据集根目录，同上
        :param transform: 数据转换器，默认为None
        """
        assert isinstance(image_root_dir, list)
        assert isinstance(mask_root_dir, list)

        self.image_root_dir = image_root_dir
        self.mask_root_dir = mask_root_dir
        self.transform = transform
        self.image_path = []
        self.mask_path = []

        for root_dir in image_root_dir:
            for image_name in os.listdir(root_dir):
                image_path = os.path.join(root_dir, image_name)
                self.image_path.append(image_path)

        for mask_dir in mask_root_dir:
            for mask_name in os.listdir(mask_dir):
                mask_path = os.path.join(mask_dir, mask_name)
                self.mask_path.append(mask_path)

    # 获取数据集的大小
    def __len__(self):
        return len(self.image_path)

    # 根据索引获取数据集中的某个元素
    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = Image.open(image_path).convert('RGB')

        mask_path = self.mask_path[index]
        mask_image = Image.open(mask_path)
        mask = tf.to_tensor(mask_image)

        if self.transform:
            image = self.transform(image)

        return image, mask

    # 获取mask的unique值
    def get_unique_mask(self):
        unique_mask = []
        for index in range(self.__len__()):
            _, mask = self.__getitem__(index)
            unique = torch.unique(mask)
            for item in unique:
                unique_mask.append(item)

        unique_mask = torch.tensor(unique_mask)
        return torch.unique(unique_mask)


# 创建数据集
def get_dataset(image_root_dir, mask_root_dir, batch_size=64, transform=None, shuffle=True):
    dataset = LoveDA_Dataset(image_root_dir, mask_root_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset, loader


# 验证网络并将生成的语义分割图存储在其他文件夹中，并获取评价指标
def get_segmentation(model, data_loader, save_dir, color_map, option):
    """
    :param model: 模型
    :param data_loader: 数据集
    :param save_dir: 保存路径，由列表组成，其中第一个元素表示真实掩码的存储路径，第二个元素表示预测掩码的存储路径
    :param color_map: 颜色映射
    :param option: 采样倍数
    """
    for root in save_dir:
        if not os.path.isdir(root):
            os.mkdir(root)

    to_pil = torchvision.transforms.ToPILImage()

    num_image = 0
    for (x, y) in data_loader:
        # x为原始image，y为真实掩码
        (x, y) = (x.to('cuda:0'), y.to('cuda:0'))

        for image, mask in zip(x, y):
            # 获取预测图
            predict = model(image, option).argmax(dim=0, keepdim=True)

            # 获取分割图
            predict = to_pil(color_map[predict].squeeze(0).permute(2, 0, 1))

            # 获取保存路径
            save_predict = os.path.join(save_dir[0], str(num_image)+'.png')
            num_image += 1

            # 保存图片
            predict.save(save_predict)


# 裁剪图像并进行移动
def Crop_images(images_root_dir, masks_root_dir, patch_size, color_map, mask_rgb_dir, mask_des_dir,
                to_tensor=get_totensor(), to_pil=get_topil()):
    """
    :param mask_des_dir: 分割掩码存储路径
    :param mask_rgb_dir: rgb掩码存储路径
    :param images_root_dir: 图像源路径
    :param masks_root_dir: 掩码源路径
    :param patch_size: 分割图像大小
    :param color_map: 颜色映射
    :param to_tensor: 图像-张量转换器
    :param to_pil: 张量-图像转换器
    """
    # 若目标路径不存在，则创建目标路径
    for mask_path in mask_rgb_dir:
        if not os.path.isdir(mask_path):
            os.mkdir(mask_path)

    image_type = 0  # 0表示Rural, 1表示urban
    # 读取每一张图片并进行裁剪和移动操作
    for image_dir, mask_dir in zip(images_root_dir, masks_root_dir):
        # 同时读取图片和掩码
        for image_name, mask_name in zip(os.listdir(image_dir), os.listdir(mask_dir)):
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, mask_name)

            # 打开图片和掩码
            rural_image = Image.open(image_path)
            rural_mask = Image.open(mask_path)

            # 获取image的大小
            image_size = rural_image.size

            assert image_size[0] % patch_size[0] == 0
            assert image_size[1] % patch_size[1] == 0

            num_height = image_size[0] // patch_size[0]
            num_width = image_size[1] // patch_size[1]

            max_pooling = nn.MaxPool2d(kernel_size=(num_height - 1, num_width - 1), stride=(num_height, num_width),
                                       padding=1)

            # 对掩码图进行处理
            mask = to_tensor(rural_mask)
            cropped_mask = max_pooling(mask)

            # 将处理后的掩码图从tensor转化为灰度图像
            gray_mask = to_pil(cropped_mask.squeeze(0))

            # 对处理后的掩码图进行color map操作
            cropped_mask = (cropped_mask // 0.0039).long()

            rgb_cropped_mask = color_map[cropped_mask].squeeze(0).permute(2, 0, 1)

            # 将处理后的掩码图及其rgb形式保存到文件夹中
            save_rgb = os.path.join(mask_rgb_dir[image_type], image_name)
            save_mask = os.path.join(mask_des_dir[image_type], image_name)

            rgb_cropped_mask = to_pil(rgb_cropped_mask)

            rgb_cropped_mask.save(save_rgb)
            gray_mask.save(save_mask)

        image_type += 1


def get_loader_rgbmask(data_loader, color_map, des_path):
    """
    :param data_loader: 获取数据
    :param color_map: 颜色映射
    :param des_path: 目标路径
    """
    num_features = 0
    to_pil = get_topil()
    for (_, y) in data_loader:
        for item in y:
            rgb_item = to_pil(color_map[(item // 0.0039).long()].squeeze(0).permute(2, 0, 1))
            path = os.path.join(des_path, str(num_features)+'.png')
            rgb_item.save(path)
            num_features += 1
