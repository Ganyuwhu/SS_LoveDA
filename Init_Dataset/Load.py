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

import torchvision.transforms.functional as tf
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image


# 获取transform
def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor()
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

        print('Success!')

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
def get_dataset(image_root_dir, mask_root_dir, batch_size=64, transform=None):
    dataset = LoveDA_Dataset(image_root_dir, mask_root_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, loader
