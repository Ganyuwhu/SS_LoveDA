#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
import Nets.Precision as Pre


def loss_fun(predict, y):
    assert(predict.shape == y.shape)
    diff = predict**2 - y**2
    result = diff.sum() / (1024*1024)
    return result


def test(model, data_loader, learning_rate, decay, epochs, scheduler, option=None, loss_fn=nn.CrossEntropyLoss()):
    """
    :param model: 模型
    :param data_loader: 数据加载器
    :param learning_rate: 学习率
    :param loss_fn: 损失函数
    :param decay: 正则化常数
    :param epochs: 迭代次数
    :param scheduler: 学习率迭代器
    :param option: 模型类型
    :return: 训练完毕的model
    """

    lr = learning_rate

    for epoch in range(epochs):
        # 定义优化器
        optimizer = opt.Adam(model.parameters(), lr=lr, weight_decay=decay)

        loss_item = []
        for(x, y) in data_loader:
            loss_item = []

            (x, y) = (x.to('cuda:0'), y.to('cuda:0'))
            optimizer.zero_grad()
            if option is not None:
                predict = model(x, option).to('cuda:0')  # 此时输出的为(batch_size, 8, 1024, 1024)
            else:
                predict = model(x, option).to('cuda:0')  # 此时输出的为(batch_size, 8, 1024, 1024)
            y = (y.squeeze(1) // 0.0039).long()

            loss = loss_fn(predict, y)
            loss_item.append(loss.item())

            loss.backward()
            optimizer.step()

        print(
            f'第{epoch+1}次训练完毕，损失函数值为{np.average(loss_item)}'
        )

        # if np.average(loss_item) < 0.2:
        #     break

        if scheduler is not None:
            lr = scheduler()

    return model


def get_precisions(model, train_loader, option):
    pa = []
    mean_pa = []
    mIoU = []
    Fw_IoU = []
    K = []
    F1_score = []

    for (x, y) in train_loader:
        (x, y) = (x.to('cuda:0'), (y.to('cuda:0') // 0.0039).long())

        predict = model(x, option=option).argmax(dim=1, keepdim=True)

        pa.append(Pre.get_pa(predict, y))
        mean_pa.append(Pre.get_mean_pa(predict, y))
        mIoU.append(Pre.get_mIoU(predict, y))
        Fw_IoU.append(Pre.get_FWIoU(predict, y))
        K.append(Pre.get_K(predict, y))
        F1_score.append(Pre.get_f1score(predict, y))

    print(f"PA：{np.average(pa)}")
    print(f"MPA：{np.average(mean_pa)}")
    print(f"mIoU：{np.average(mIoU)}")
    print(f"FW_IoU：{np.average(Fw_IoU)}")
    print(f"K：{np.average(K)}")
    print(f"F1_score：{np.average(F1_score)}")


def count_parameters(model):
    """
    计算模型中的参数规模
    :param model: 模型
    :return: 模型中的参数数目
    """
    parameters = []
    for p in model.parameters():
        if p.requires_grad:
            parameters.append(p.numel())

    return np.sum(parameters)
