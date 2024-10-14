import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import Init_Dataset.Load as Load
import Nets.test as test

from Nets import FCN
from Nets import HRNet
from Nets import PSPNet
from Nets import Precision
from Nets import UNet

# 源数据地址
train_image_root_dir = ['E:\\gzr\\SS_LoveDA\\Initial\\Train\\Rural\\images_png',
                        'E:\\gzr\\SS_LoveDA\\Initial\\Train\\Urban\\images_png']

train_mask_root_dir = ['E:\\gzr\\SS_LoveDA\\Initial\\Train\\Rural\\masks_png',
                       'E:\\gzr\\SS_LoveDA\\Initial\\Train\\Urban\\masks_png']

val_image_root_dir = ['E:\\gzr\\SS_LoveDA\\Initial\\Val\\Rural\\images_png',
                      'E:\\gzr\\SS_LoveDA\\Initial\\Val\\Urban\\images_png']

val_mask_root_dir = ['E:\\gzr\\SS_LoveDA\\Initial\\Val\\Rural\\masks_png',
                     'E:\\gzr\\SS_LoveDA\\Initial\\Val\\Urban\\masks_png']

test_image_root_dir = ['E:\\gzr\\SS_LoveDA\\Initial\\Test\\Rural\\images_png',
                       'E:\\gzr\\SS_LoveDA\\Initial\\Test\\Urban\\images_png']

# 训练结果保存地址
save_train_dir_8 = ['E:\\gzr\\SS_LoveDA\\Results\\Train\\Predict_8',
                    'E:\\gzr\\SS_LoveDA\\Results\\Train\\True_8']

save_test_dir_8 = ['E:\\gzr\\SS_LoveDA\\Results\\Test\\Predict_8',
                   'E:\\gzr\\SS_LoveDA\\Results\\Test\\True_8']

save_val_dir_8 = ['E:\\gzr\\SS_LoveDA\\Results\\Val\\Predict_8',
                  'E:\\gzr\\SS_LoveDA\\Results\\Val\\True_8']

save_train_dir_16 = ['E:\\gzr\\SS_LoveDA\\Results\\Train\\Predict_16',
                     'E:\\gzr\\SS_LoveDA\\Results\\Train\\True_16']

save_test_dir_16 = ['E:\\gzr\\SS_LoveDA\\Results\\Test\\Predict_16',
                    'E:\\gzr\\SS_LoveDA\\Results\\Test\\True_16']

save_val_dir_16 = ['E:\\gzr\\SS_LoveDA\\Results\\Val\\Predict_16',
                   'E:\\gzr\\SS_LoveDA\\Results\\Val\\True_16']

save_train_dir_32 = ['E:\\gzr\\SS_LoveDA\\Results\\Train\\Predict_32',
                     'E:\\gzr\\SS_LoveDA\\Results\\Train\\True_32']

save_test_dir_32 = ['E:\\gzr\\SS_LoveDA\\Results\\Test\\Predict_32',
                    'E:\\gzr\\SS_LoveDA\\Results\\Test\\True_32']

save_val_dir_32 = ['E:\\gzr\\SS_LoveDA\\Results\\Val\\Predict_32',
                   'E:\\gzr\\SS_LoveDA\\Results\\Val\\True_32']

save_train_dir_hr = ['E:\\gzr\\SS_LoveDA\\Results\\Train\\Predict_hr',
                     'E:\\gzr\\SS_LoveDA\\Results\\Train\\True_hr']

# 分割掩码保存地址
cropped_train_rgb_mask_dir = ["E:\\gzr\\SS_LoveDA\\Cropped\\Train\\Rural\\rgb_masks_png",
                              "E:\\gzr\\SS_LoveDA\\Cropped\\Train\\Urban\\rgb_masks_png"]

cropped_val_rgb_mask_dir = ["E:\\gzr\\SS_LoveDA\\Cropped\\Val\\Rural\\rgb_masks_png",
                            "E:\\gzr\\SS_LoveDA\\Cropped\\Val\\Urban\\rgb_masks_png"]

cropped_train_mask_dir = ["E:\\gzr\\SS_LoveDA\\Cropped\\Train\\Rural\\masks_png",
                          "E:\\gzr\\SS_LoveDA\\Cropped\\Train\\Urban\\masks_png"]

cropped_val_mask_dir = ["E:\\gzr\\SS_LoveDA\\Cropped\\Val\\Rural\\masks_png",
                        "E:\\gzr\\SS_LoveDA\\Cropped\\Val\\Urban\\masks_png"]

# 语义掩码与获取颜色映射
mask_unique = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)

color_map = torch.tensor([
    [1.0, 0.0, 0.0],  # 红色
    [0.0, 1.0, 0.0],  # 绿色
    [0.0, 0.0, 1.0],  # 蓝色
    [1.0, 1.0, 0.0],  # 黄色
    [1.0, 0.0, 1.0],  # 品红色
    [0.0, 1.0, 1.0],  # 青色
    [0.5, 0.5, 0.5],  # 灰色
    [1.0, 1.0, 1.0],  # 白色
], dtype=torch.float32).to('cuda:0')

mask_unique = mask_unique.to('cuda:0')

########################################################################################################################
# 分割图像与创loader
# 分割训练集图像
# Load.Crop_images(images_root_dir=train_image_root_dir, masks_root_dir=train_mask_root_dir, patch_size=(256, 256),
#                  color_map=color_map, mask_rgb_dir=cropped_train_rgb_mask_dir, mask_des_dir=cropped_train_mask_dir)

# # 分割验证集图像
# Load.Crop_images(images_root_dir=val_image_root_dir, masks_root_dir=val_mask_root_dir, patch_size=(256, 256),
#                  color_map=color_map, images_des_dir=cropped_val_images_dir,
#                  masks_des_dir=cropped_val_masks_dir, mask_rgb_dir=cropped_val_rgb_masks_dir)

# 创建loader
transform = Load.get_totensor()
_, train_loader = Load.get_dataset(train_image_root_dir, cropped_train_mask_dir, batch_size=100, transform=transform)

# Load.get_loader_rgbmask(train_loader, color_map, des_path='E:\\gzr\\SS_LoveDA\\Results\\Train\\Train_loader_masks')

########################################################################################################################

########################################################################################################################
# 初始化网络以及训练
FCN_8 = torch.load('FCN_8_cropped.pth').to('cuda:0')
FCN_16 = torch.load('FCN_16_cropped.pth').to('cuda:0')
FCN_32 = torch.load('FCN_32_cropped.pth').to('cuda:0')
HR_Net = HRNet.HRNet().to('cuda:0')

print('FCN_8训练结果：')
FCN_8 = test.test(FCN_8, train_loader, learning_rate=0.0001, decay=0, epochs=100, option=8)
torch.save(FCN_8, 'FCN_8_cropped.pth')

print('FCN_16训练结果：')
FCN_16 = test.test(FCN_16, train_loader, learning_rate=0.0001, decay=0, epochs=100, option=16)
torch.save(FCN_16, 'FCN_16_cropped.pth')

print('FCN_32训练结果：')
FCN_32 = test.test(FCN_32, train_loader, learning_rate=0.0001, decay=0, epochs=100, option=32)
torch.save(FCN_32, 'FCN_32_cropped.pth')

print('HR_Net训练结果')
HR_Net = test.test(HR_Net, train_loader, learning_rate=0.00025, decay=0, epochs=200, option=None)
torch.save(HR_Net, 'HRNet.pth')

########################################################################################################################

########################################################################################################################
# 读取网络和测试
FCN_8 = torch.load('FCN_8_cropped.pth')
FCN_16 = torch.load('FCN_16_cropped.pth')
FCN_32 = torch.load('FCN_32_cropped.pth')
HR_Net = torch.load('HRNet.pth')

Load.get_segmentation(FCN_8, train_loader, save_train_dir_8, color_map, option=8)
Load.get_segmentation(FCN_16, train_loader, save_train_dir_16, color_map, option=16)
Load.get_segmentation(FCN_32, train_loader, save_train_dir_32, color_map, option=32)
Load.get_segmentation(HR_Net, train_loader, save_train_dir_hr, color_map, option=None)
########################################################################################################################

