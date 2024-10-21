#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

"""
    语义分割的精度测试
"""


# Focal loss函数
class Focal_loss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super(Focal_loss, self).__init__()
        """
        Focal Loss函数
        :param gamma: 用于调整难易样本的权重，默认为2
        :param reduction: 指定输出的方式，默认为求平均值
        """
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predict, mask):
        """
        :param predict: 预测值，大小为(batch_size, classes, height, width)
        :param mask: 实际掩码，已经转换为one-hot编码，大小为(batch_size, 1, height, width)
        :return: loss: 损失值
        """

        # 逐像素点获取每个类别的可能性，并用mask获取实际类别的可能性
        probs = f.softmax(predict, dim=1)

        # 获取预测值和真实值的可能性
        predict = probs.argmax(dim=1, keepdim=True)  # 返回类别

        probs_predict = torch.gather(probs, dim=1, index=predict)
        probs_true = torch.gather(probs, dim=1, index=mask)

        # 计算p_t
        equal_tensor = (predict == mask).int()

        # 对probs_predict中的每个元素，若equal_tensor中的元素为1，则返回该元素，否则返回1-p
        p_t = equal_tensor * probs_predict + (1-equal_tensor) * (1-probs_predict)

        # 获取focal loss
        focal_loss = -(1-p_t) ** self.gamma * torch.log(p_t + 1e-5)

        # 根据reduction选取不同的输出项
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 语义分割评价指标

# 获取x和y之间的像素精度，输入均为tensor
def get_pa(x, y):

    count = torch.sum(x == y).item()
    pa = count / torch.numel(x)

    return pa


# 获取x和y之间的平均像素精度，输入均为tensor
def get_mean_pa(x, y, classes=8):
    """
    :param x: 输入图
    :param y: 输出图
    :param classes: 类别数，等于数据集的语义种类数
    :return: 平均pa
    """

    precisions = []
    for i in range(classes):
        y_i = torch.sum(y == i).item()
        x_y = torch.sum((x == y) & (y == i)).item()

        if y_i == 0 & x_y == 0:
            precisions.append(1)
        elif y_i == 0 & x_y != 0:
            precisions.append(0)
        else:
            precisions.append(x_y / y_i)

    return np.average(precisions)


# 获取平均交并比
def get_mIoU(x, y, classes=8):

    IoU = []

    for i in range(classes):
        x_i = torch.sum(x == i).item()
        y_i = torch.sum(y == i).item()
        x_y = torch.sum((x == i) & (y == i)).item()

        if x_i + y_i - x_y == 0:
            IoU.append(1)
        else:
            IoU.append(x_y / (x_i + y_i - x_y))

    mIoU = np.average(IoU)
    return mIoU


# 获取频率权重交并比
def get_FWIoU(x, y, classes=8):

    IoU = []
    class_count = []  # 获取每一类别的元素个数

    for i in range(classes):
        x_i = torch.sum(x == i).item()
        y_i = torch.sum(y == i).item()
        x_y = torch.sum((x == i) & (y == i)).item()

        if x_i + y_i - x_y == 0:
            IoU.append(1)

        else:
            IoU.append(x_y / (x_i + y_i - x_y))  # 求IoU

        class_count.append(y_i)

    assert np.sum(class_count) == torch.numel(y)

    for i in range(len(IoU)):
        weight = class_count[i] / torch.numel(y)
        IoU[i] = weight * IoU[i]

    fwIoU = np.sum(IoU)

    return fwIoU


# 获取F1-score
def get_f1score(x, y, classes=8):
    # 首先获取pa
    pa = get_pa(x, y)

    # 获取平均召回率
    ra = []
    for i in range(classes):
        # 获取实际掩码为第i类的pixel总数
        y_i = torch.sum(y == i).item()

        # 获取被召回元素的个数
        x_y = torch.sum((x == y) & (y == i)).item()

        if y_i == 0:
            ra.append(1)
        else:
            ra.append(x_y / y_i)

    mean_ra = np.average(ra)

    f1_score = 2 / (1/pa + 1/mean_ra)

    return f1_score


# 获取Kappa Coefficient
def get_K(x, y, classes=8):

    p0 = get_pa(x, y)

    # 获取每个类别的预测数和实际数
    X = []
    Y = []

    for i in range(classes):
        X.append(torch.sum(x == i).item())
        Y.append(torch.sum(y == i).item())

    assert np.sum(X) == np.sum(Y) == torch.numel(y)

    count = 0
    for item_x, item_y in zip(X, Y):
        count += item_x * item_y

    pe = count / (torch.numel(y) ** 2)

    Kappa = (p0 - pe) / (1 - pe)
    return Kappa
