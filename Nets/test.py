#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
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


def test(model, data_loader, learning_rate, decay, epochs, option=None, loss_fn=nn.CrossEntropyLoss()):
    """
    :param model: 模型
    :param data_loader: 数据加载器
    :param learning_rate: 学习率
    :param loss_fn: 损失函数
    :param decay: 正则化常数
    :param epochs: 迭代次数
    :param option: 模型类型
    :return: 训练完毕的model
    """

    # 定义优化器
    optimizer = opt.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

    for epoch in range(epochs):
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
            f'第{epoch+1}次训练完毕，损失函数值为{numpy.average(loss_item)}'
        )

    return model
