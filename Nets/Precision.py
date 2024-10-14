#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as f

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

# 0、获取混淆矩阵

# 1、PA

# 2、MPA

# 3、mIoU

# 4、FWIoU

# 5、K
