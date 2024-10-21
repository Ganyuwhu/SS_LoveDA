#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import math


class FactorScheduler:
    # 构建单因子调度器
    def __init__(self, factor=0.99, stop_factor=1e-7, base_lr=0.1):
        """
        :param self: 指向调度器本身
        :param factor: 每次衰减的倍率
        :param stop_factor: 学习率的下界，保证学习率不会过低导致训练过慢
        :param base_lr: 初始学习率
        """
        self.factor = factor
        self.stop_factor = stop_factor
        self.current_lr = base_lr

    def __call__(self):
        self.current_lr = max(self.stop_factor, self.current_lr * self.factor)
        return self.current_lr


class OriginFactor:
    # 构建普通调度器
    def __init__(self, result=0, result_temp=0, stop_factor=1e-10, base_lr=0.1):
        """
        :param result: 存放当前epoch的损失函数值
        :param result_temp: 存放上一epoch的损失函数值
        :param stop_factor: 学习率的下界，保证学习率不会过低导致训练过慢
        :param base_lr: 当前学习率
        """
        self.result = result
        self.result_temp = result_temp
        self.stop_factor = stop_factor
        self.current_lr = base_lr

    def __call__(self, result):
        """
        :param result: 当前epoch的损失函数值
        :return: self.current_lr
        """
        # 更新scheduler中的损失函数值
        self.result_temp = self.result
        self.result = result

        # 判断是否降低学习率
        if abs((self.result_temp-self.result-1e-5)/(self.result_temp+1e-10)) < 0.02:
            self.current_lr /= 10
        return max(self.current_lr, self.stop_factor)


class CosineScheduler:
    def __init__(self, base_lr, final_lr=1e-10, warm_up=50):
        """
        :param base_lr: 初始学习率
        :param final_lr: 目标学习率
        :param warm_up: 目标epoch，epoch高于这个值会将lr固定为final_lr
        """
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warm_up = warm_up

    def __call__(self, epoch):
        """
        :param epoch: 输入当前epoch
        :return: 计算出的下一个lr的值
        """
        if epoch < self.warm_up:
            lr = self.final_lr + (self.base_lr - self.final_lr) * (1 + math.cos(math.pi * epoch / self.warm_up))
        else:
            lr = self.final_lr

        return lr


class MultiFactorScheduler:
    def __init__(self, factor=0.5, milestone=None, base_lr=0.01):
        """
        :param factor: 学习率的下降率
        :param milestone: 用于存储需要下降学习率的epoch的tuple
        :param base_lr: 初始学习率
        """
        if milestone is None:
            milestone = [10, 20, 30, 40, 50]
        self.factor = factor
        self.milestone = milestone
        self.current_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.milestone:
            self.current_lr *= self.factor
        return self.current_lr


class Nochange_Scheduler:
    def __init__(self, base_lr=0.01):
        """
        :param base_lr: 初始学习率
        """
        self.lr = base_lr

    def __call__(self):
        return self.lr


def init_scheduler(base_lr, Type='Origin'):
    if Type == 'Origin':
        scheduler = OriginFactor(base_lr=base_lr)
    elif Type == 'Factor':
        scheduler = FactorScheduler(base_lr=base_lr)
    elif Type == 'Cosine':
        scheduler = CosineScheduler(base_lr=base_lr)
    elif Type == 'Multi':
        scheduler = MultiFactorScheduler(base_lr=base_lr)
    elif Type == 'None':
        scheduler = Nochange_Scheduler(base_lr=base_lr)
    else:
        raise ValueError('Type must be chosen from {Origin, Factor}')
    return scheduler
