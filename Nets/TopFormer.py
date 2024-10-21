#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as f


# 创建下采样层，下采样层的结构同HRNet一致
class Down_2(nn.Module):
    # 构建2倍下采样块
    def __init__(self, in_channels, out_channels):
        super(Down_2, self).__init__()

        assert out_channels % 2 == 0

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),

            # BottleNeck块
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=out_channels // 2, out_channels=2 * out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x)


class Down_4(nn.Module):
    # 构建4倍下采样块
    def __init__(self, in_channels, out_channels):
        super(Down_4, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),

            # BottleNeck块
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x)


class Down_8(nn.Module):
    # 构建8倍下采样块
    def __init__(self, in_channels, out_channels):
        super(Down_8, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4*out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=4*out_channels, out_channels=2*out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),

            # BottleNeck块
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x)


class Down_16(nn.Module):
    # 构建16倍下采样块
    def __init__(self, in_channels, out_channels):
        super(Down_16, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8*out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=8*out_channels, out_channels=4*out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=4*out_channels, out_channels=2*out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),

            # BottleNeck块
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x)


# 基于缩放点积注意力的多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, **kwargs):
        """
        :param d_model: 多头注意力对应的隐藏层单元数
        :param heads: 多头注意力中头的数量
        :param kwargs: 其他参数
        """
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.heads = heads
        self.d = d_model // heads
        self.scale_factor = 1.0 / (self.d**0.5)

        # 使用全连接层定义各个线性变换
        self.query_layer = nn.Linear(d_model, self.d)
        self.key_layer = nn.Linear(d_model, self.d)
        self.value_layer = nn.Linear(d_model, self.d)
        self.get_query = nn.ModuleList()
        self.get_key = nn.ModuleList()
        self.get_value = nn.ModuleList()
        for i in range(self.heads):
            self.get_query.append(nn.Linear(self.d, self.d))
            self.get_key.append(nn.Linear(self.d, self.d))
            self.get_value.append(nn.Linear(self.d, self.d))

        self.fc_out = nn.Linear(self.heads*self.d, self.d_model, bias=False)

    def forward(self, Q, K, V, mask=None):

        # 第一步，获取键、值、查询
        batch_size = K.shape[0]
        T = K.shape[1]
        d = K.shape[2]

        # 第二步，将h个头的键、值、查询进行整合
        cat_K = torch.zeros(self.heads, batch_size, T, d)
        cat_V = torch.zeros(self.heads, batch_size, T, d)
        cat_Q = torch.zeros(self.heads, batch_size, T, d)
        for i in range(self.heads):
            cat_K[i] = self.get_key[i](K)
            cat_V[i] = self.get_value[i](K)
            cat_Q[i] = self.get_query[i](K)

        cat_K = cat_K.reshape(batch_size, T, self.heads, d)
        cat_V = cat_V.reshape(batch_size, T, self.heads, d)
        cat_Q = cat_Q.reshape(batch_size, T, self.heads, d)

        # 第三步，利用torch.einsum进行批处理，计算其注意力评分函数
        attention_score = torch.einsum("bqhd, bkhd -> bhqk", [cat_Q, cat_K])
        # 由于一个字母不能在箭头右侧出现两次，因此用q和k代替T

        # 掩蔽操作
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, 0)

        # 第四步，利用f.softmax函数计算注意力权重，由于是对每个head单独计算权重，因此需要指定dim=-1
        attention_weight = f.softmax(attention_score*self.scale_factor, dim=-1)

        # 第五步，计算将每个注意力汇聚concat后的输出
        out = torch.einsum("bhqk, bkhd -> bhqd", [attention_weight, cat_V]).to('cuda:0')
        out = out.reshape(batch_size, T, self.heads*d)

        # 第六步，通过全连接层得到输出
        output = self.fc_out(out)

        return output


# 逐位前馈神经网络
class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):  # hidden_dim应远大于input_dim
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(float(input_dim)/hidden_dim)

    def forward(self, x):
        out = self.fc2(self.dropout(self.relu(self.fc1(x))))
        return out


# 构建Top模块
class TokenPyramidModule(nn.Module):
    def __init__(self):
        super(TokenPyramidModule, self).__init__()

        # 预处理，降维
        self.Preprocess = nn.AvgPool2d(kernel_size=3, stride=4, padding=1)

        # Pooling层，数字代表第几层的Pooling
        self.Pool1 = nn.AvgPool2d(kernel_size=7, stride=16, padding=1)
        self.Pool2 = nn.AvgPool2d(kernel_size=3, stride=8, padding=1)
        self.Pool3 = nn.AvgPool2d(kernel_size=2, stride=4, padding=0)

        # 利用卷积层进行下采样
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.Conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.Conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.Conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        :param x: 输入图
        :return: 拼接图feature和对应的通道数channels
        """
        Pre = self.Preprocess(x)

        conv1 = f.relu6(self.Conv1(Pre))
        conv2 = f.relu6(self.Conv2(conv1))
        conv3 = f.relu6(self.Conv3(conv2))
        conv4 = f.relu6(self.Conv4(conv3))
        conv5 = f.relu6(self.Conv5(conv4))

        feature1 = self.Pool1(conv1)
        feature2 = self.Pool2(conv2)
        feature3 = self.Pool3(conv3)
        feature4 = conv5

        feature = torch.cat((feature1, feature2, feature3, feature4), dim=1)
        local_tokens = [conv1, conv2, conv3, conv4]

        return feature, local_tokens


# 构建ViT模块
class Former(nn.Module):
    def __init__(self, input_size, heads, L):
        """
        :param input_size: 注意力模块输入大小
        :param heads: 多头注意力的heads数
        :param L: 注意力Block的个数
        """
        super(Former, self).__init__()

        self.input_size = input_size
        self.heads = heads
        self.L = L

        self.Blocks = nn.ModuleList()

        for i in range(L):
            self.Blocks.append(
                nn.Sequential(
                    MultiHeadAttention(d_model=input_size, heads=heads),
                    FFN(input_dim=input_size, hidden_dim=2*input_size)
                )
            )

    def forward(self, x):
        batch_size = x.shape[0]
        channels = x.shape[1]
        x = x.reshape(batch_size, channels, -1)

        for i in range(self.L):
            key = self.Blocks[i][0].key_layer(x)
            query = self.Blocks[i][0].query_layer(x)
            value = self.Blocks[i][0].value_layer(x)

            x = self.Blocks[i][1](self.Blocks[i][0](query, key, value))

        output = x.reshape(batch_size, channels, -1, self.input_size//self.heads)

        return output


# Semantic Injection Module
class Sim(nn.Module):
    def __init__(self, in_channels, out_channels, factor=2):
        super(Sim, self).__init__()
        # 路径1，对token进行降维和BatchNorm处理
        self.ProcessToken = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels)
        )

        # 路径2，对semantic进行降维和BatchNorm处理，该路径输出与路径1输出结合
        self.ProcessFeature1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels)
        )

        # 路径3，对semantic进行降维和BatchNorm处理，该路径输出与Hadmard积的结果直接相加
        self.ProcessFeature2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, token, semantic):
        semantic = f.interpolate(semantic, (token.shape[2], token.shape[3]), mode='bilinear', align_corners=True)

        pro_token = self.ProcessToken(token)
        pro_semantic1 = f.sigmoid(self.ProcessFeature1(semantic))
        pro_semantic2 = self.ProcessFeature2(semantic)

        return pro_token * pro_semantic1 + pro_semantic2


# Segmentation Head
class SegeHead(nn.Module):
    def __init__(self, in_channels=32, out_channels=8):
        super(SegeHead, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2*out_channels, kernel_size=1)
        self.BN = nn.BatchNorm2d(num_features=2*out_channels)
        self.Conv2 = nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, feature_maps):
        size = feature_maps[0].shape[2], feature_maps[0].shape[3]
        for i in range(len(feature_maps)):
            if i == 0:
                continue
            else:
                feature_maps[i] = f.interpolate(feature_maps[i], size=size, mode='bilinear', align_corners=True)
                feature_maps[0] += feature_maps[i]

        return self.Conv2(f.relu6(self.BN(self.Conv1(feature_maps[0]))))


# TopFormer
class TopFormer(nn.Module):
    def __init__(self, feature_shape):
        super(TopFormer, self).__init__()

        assert isinstance(feature_shape, tuple)  # feature_shape must be a tuple composed by 2 elements

        assert feature_shape[0] % 64 == 0
        assert feature_shape[1] % 64 == 0

        self.input_size = (feature_shape[0] // 64) * (feature_shape[1] // 64)
        self.heads = feature_shape[0] // 64
        self.L = 3
        self.channels = [64, 128, 256, 512]

        self.Top = TokenPyramidModule()
        self.Former = Former(input_size=self.input_size, heads=self.heads, L=self.L)

        self.Sim = nn.ModuleList()

        for i in range(len(self.channels)):
            self.Sim.append(Sim(in_channels=self.channels[i], out_channels=32))

        self.SegeHead = SegeHead()

    def forward(self, x, option=None):
        features, local_tokens = self.Top(x)
        global_semantics = self.Former(features)

        # 分割不同的global_semantic
        global_semantics = list(torch.split(global_semantics, self.channels, dim=1))

        feature_maps = []
        for i in range(len(self.channels)):
            feature_maps.append(self.Sim[i](local_tokens[i], global_semantics[i]))

        predict_map = self.SegeHead(feature_maps)

        return predict_map
