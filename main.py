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
from Nets import HRFormer
from Nets import Precision
from Nets import UNet
from Nets import Schedulers
from Nets import TopFormer
from Nets import MANet

# 源数据地址
train_image_root_dir = ['D:\\gzr_dataset\\LoveDA\\Initial\\Train\\Rural\\images_png',
                        'D:\\gzr_dataset\\LoveDA\\Initial\\Train\\Urban\\images_png']

train_mask_root_dir = ['D:\\gzr_dataset\\LoveDA\\Initial\\Train\\Rural\\masks_png',
                       'D:\\gzr_dataset\\LoveDA\\Initial\\Train\\Urban\\masks_png']

val_image_root_dir = ['D:\\gzr_dataset\\LoveDA\\Initial\\Val\\Rural\\images_png',
                      'D:\\gzr_dataset\\LoveDA\\Initial\\Val\\Urban\\images_png']

val_mask_root_dir = ['D:\\gzr_dataset\\LoveDA\\Initial\\Val\\Rural\\masks_png',
                     'D:\\gzr_dataset\\LoveDA\\Initial\\Val\\Urban\\masks_png']

test_image_root_dir = ['D:\\gzr_dataset\\LoveDA\\Initial\\Test\\Rural\\images_png',
                       'D:\\gzr_dataset\\LoveDA\\Initial\\Test\\Urban\\images_png']

# 训练结果保存地址
save_train_dir_8 = ['D:\\gzr_dataset\\LoveDA\\Results\\Train\\Predict_8',
                    'D:\\gzr_dataset\\LoveDA\\Results\\Train\\True_8']

save_test_dir_8 = ['D:\\gzr_dataset\\LoveDA\\Results\\Test\\Predict_8',
                   'D:\\gzr_dataset\\LoveDA\\Results\\Test\\True_8']

save_val_dir_8 = ['D:\\gzr_dataset\\LoveDA\\Results\\Val\\Predict_8',
                  'D:\\gzr_dataset\\LoveDA\\Results\\Val\\True_8']

save_train_dir_16 = ['D:\\gzr_dataset\\LoveDA\\Results\\Train\\Predict_16',
                     'D:\\gzr_dataset\\LoveDA\\Results\\Train\\True_16']

save_test_dir_16 = ['D:\\gzr_dataset\\LoveDA\\Results\\Test\\Predict_16',
                    'D:\\gzr_dataset\\LoveDA\\Results\\Test\\True_16']

save_val_dir_16 = ['D:\\gzr_dataset\\LoveDA\\Results\\Val\\Predict_16',
                   'D:\\gzr_dataset\\LoveDA\\Results\\Val\\True_16']

save_train_dir_32 = ['D:\\gzr_dataset\\LoveDA\\Results\\Train\\Predict_32',
                     'D:\\gzr_dataset\\LoveDA\\Results\\Train\\True_32']

save_test_dir_32 = ['D:\\gzr_dataset\\LoveDA\\Results\\Test\\Predict_32',
                    'D:\\gzr_dataset\\LoveDA\\Results\\Test\\True_32']

save_val_dir_32 = ['D:\\gzr_dataset\\LoveDA\\Results\\Val\\Predict_32',
                   'D:\\gzr_dataset\\LoveDA\\Results\\Val\\True_32']

save_train_dir_hr = ['D:\\gzr_dataset\\LoveDA\\Results\\Train\\Predict_hr',
                     'D:\\gzr_dataset\\LoveDA\\Results\\Train\\True_hr']

save_train_dir_psp = ['D:\\gzr_dataset\\LoveDA\\Results\\Train\\Predict_psp',
                      'D:\\gzr_dataset\\LoveDA\\Results\\Train\\Predict_psp']

save_trian_dir_unet = ['D:\\gzr_dataset\\LoveDA\\Results\\Train\\Predict_unet',
                       'D:\\gzr_dataset\\LoveDA\\Results\\Train\\Predict_unet']

save_train_dir_top = ['D:\\gzr_dataset\\LoveDA\\Results\\Train\\Predict_top',
                      'D:\\gzr_dataset\\LoveDA\\Results\\Train\\Predict_top']

save_train_dir_ma = ['D:\\gzr_dataset\\LoveDA\\Results\\Train\\Predict_ma',
                     'D:\\gzr_dataset\\LoveDA\\Results\\Train\\Predict_ma']

# 分割掩码保存地址
cropped_train_rgb_mask_dir = ["D:\\gzr_dataset\\LoveDA\\Cropped\\Train\\Rural\\rgb_masks_png",
                              "D:\\gzr_dataset\\LoveDA\\Cropped\\Train\\Urban\\rgb_masks_png"]

cropped_val_rgb_mask_dir = ["D:\\gzr_dataset\\LoveDA\\Cropped\\Val\\Rural\\rgb_masks_png",
                            "D:\\gzr_dataset\\LoveDA\\Cropped\\Val\\Urban\\rgb_masks_png"]

cropped_train_mask_dir = ["D:\\gzr_dataset\\LoveDA\\Cropped\\Train\\Rural\\masks_png",
                          "D:\\gzr_dataset\\LoveDA\\Cropped\\Train\\Urban\\masks_png"]

cropped_val_mask_dir = ["D:\\gzr_dataset\\LoveDA\\Cropped\\Val\\Rural\\masks_png",
                        "D:\\gzr_dataset\\LoveDA\\Cropped\\Val\\Urban\\masks_png"]

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
# 分割图像与创建loader
# 分割训练集图像
# Load.Crop_images(images_root_dir=train_image_root_dir, masks_root_dir=train_mask_root_dir, patch_size=(256, 256),
#                  color_map=color_map, mask_rgb_dir=cropped_train_rgb_mask_dir, mask_des_dir=cropped_train_mask_dir)

# # 分割验证集图像
# Load.Crop_images(images_root_dir=val_image_root_dir, masks_root_dir=val_mask_root_dir, patch_size=(256, 256),
#                  color_map=color_map, images_des_dir=cropped_val_images_dir,
#                  masks_des_dir=cropped_val_masks_dir, mask_rgb_dir=cropped_val_rgb_masks_dir)

# 创建loader
transform = Load.get_totensor()
_, train_loader = Load.get_dataset(train_image_root_dir, cropped_train_mask_dir, batch_size=16, transform=transform,
                                   shuffle=True)

_, train_loader_8 = Load.get_dataset(train_image_root_dir, cropped_train_mask_dir, batch_size=8, transform=transform,
                                    shuffle=True)

_, train_loader_10 = Load.get_dataset(train_image_root_dir, cropped_train_mask_dir, batch_size=10, transform=transform,
                                      shuffle=True)

_, train_no_shuffle = Load.get_dataset(train_image_root_dir, cropped_train_mask_dir, batch_size=16, transform=transform,
                                       shuffle=False)

# Load.get_loader_rgbmask(train_no_shuffle, color_map,
#                         des_path='D:\\gzr_dataset\\LoveDA\\Results\\Train\\Train_loader_masks')

########################################################################################################################

########################################################################################################################
# 初始化网络以及训练
# FCN_8 = torch.load('FCN_8_cropped.pth').to('cuda:0')
# FCN_16 = torch.load('FCN_16_cropped.pth').to('cuda:0')
# FCN_32 = torch.load('FCN_32_cropped.pth').to('cuda:0')
# HR_Net = torch.load('HRNet.pth').to('cuda:0')
PSP_Net = PSPNet.PSPNet().to('cuda:0')
U_Net = UNet.UNet().to('cuda:0')
Top_Former = TopFormer.TopFormer(feature_shape=(1024, 1024)).to('cuda:0')
MA_Net = torch.load('MANet.pth').to('cuda:0')
# HR_Former_T = HRFormer.HRFormer(model_type='T').to('cuda:0')


# print('FCN_8训练结果：')
# FCN_8 = test.test(FCN_8, train_loader, scheduler=Schedulers.FactorScheduler(base_lr=1e-4),
#                   learning_rate=1e-6, decay=0, epochs=10, option=8)
# torch.save(FCN_8, 'FCN_8_cropped.pth')
# #
# print('FCN_16训练结果：')
# FCN_16 = test.test(FCN_16, train_loader, scheduler=None,
#                    learning_rate=1e-6, decay=0, epochs=10, option=16)
# torch.save(FCN_16, 'FCN_16_cropped.pth')
#
# print('FCN_32训练结果：')
# FCN_32 = test.test(FCN_32, train_loader, scheduler=Schedulers.FactorScheduler(base_lr=1e-4),
#                    learning_rate=1e-6, decay=0, epochs=10, option=32)
# torch.save(FCN_32, 'FCN_32_cropped.pth')
#
# print('HR_Net训练结果')
# HR_Net = test.test(HR_Net, train_loader, scheduler=None,
#                    learning_rate=1e-4, decay=0, epochs=200, option=None)
# torch.save(HR_Net, 'HRNet.pth')
#
print('PSP_Net训练结果')
PSP_Net = test.test(PSP_Net, train_loader_8, scheduler=None,
                    learning_rate=3e-4, decay=0, epochs=150, option=None)
torch.save(PSP_Net, 'PSPNet.pth')

print('U_Net训练结果')
U_Net = test.test(U_Net, train_loader, scheduler=None,
                  learning_rate=1e-4, decay=0, epochs=150, option=None)
torch.save(U_Net, 'UNet.pth')

print('TopFormer训练结果')
Top_Former = test.test(Top_Former, train_loader, scheduler=None,
                       learning_rate=1e-4, decay=0, epochs=150, option=None)
torch.save(Top_Former, 'Topformer.pth')

print('MANet训练结果')
MA_Net = test.test(MA_Net, train_loader, scheduler=None,
                   learning_rate=3e-4, decay=0, epochs=150, option=None)
torch.save(MA_Net, 'MANet.pth')

# print('HRFormer-T的训练结果')
# HR_Former_T = test.test(HR_Former_T, train_loader_8, scheduler=None,
#                         learning_rate=1e-4, decay=0, epochs=100, option=None)
# torch.save(HR_Former_T, 'HR_Former_T.pth')


########################################################################################################################

########################################################################################################################
# 读取网络和测试
# FCN_8 = torch.load('FCN_8_cropped.pth').to('cuda:0')
# FCN_16 = torch.load('FCN_16_cropped.pth').to('cuda:0')
# FCN_32 = torch.load('FCN_32_cropped.pth').to('cuda:0')
# HR_Net = torch.load('HRNet.pth')
# PSP_Net = torch.load('PSPNet.pth')
# MA_Net = torch.load('MANet.pth')

# Load.get_segmentation(FCN_8, train_no_shuffle, save_train_dir_8, color_map, option=8)
# Load.get_segmentation(FCN_16, train_no_shuffle, save_train_dir_16, color_map, option=16)
# Load.get_segmentation(FCN_32, train_no_shuffle, save_train_dir_32, color_map, option=32)
# Load.get_segmentation(HR_Net, train_no_shuffle, save_train_dir_hr, color_map, option=None)
########################################################################################################################
# 获取参数规模
# print(f'FCN_8的参数规模为：{test.count_parameters(FCN_8)}')
# print(f'FCN_16的参数规模为：{test.count_parameters(FCN_16)}')
# print(f'FCN_32的参数规模为：{test.count_parameters(FCN_32)}')
# print(f'HRNet的参数规模为：{test.count_parameters(HR_Net)}')
# print(f'PSPNet的参数规模为：{test.count_parameters(PSP_Net)}')
# print(f'UNet的参数规模为：{test.count_parameters(U_Net)}')
# print(f'TopFormer的参数规模为：{test.count_parameters(Top_Former)}')
# print(f'MANet的参数规模为：{test.count_parameters(MA_Net)}')
# print(f'HRFormer-T的参数规模为：{test.count_parameters(HR_Former_T)}')
########################################################################################################################
# 获取精度，使用未经shuffle的数据集
# print('FCN_8的测试精度:'), test.get_precisions(FCN_8, train_no_shuffle, option=8)
# print('FCN_16的测试精度:'), test.get_precisions(FCN_16, train_no_shuffle, option=16)
# print('FCN_32的测试精度:'), test.get_precisions(FCN_32, train_no_shuffle, option=32)
# print('HRNet的测试精度:'), test.get_precisions(HR_Net, train_no_shuffle, option=None)
# print('PSPNet的测试精度:'), test.get_precisions(PSP_Net, train_no_shuffle, option=None)
# print('UNet的测试精度:'), test.get_precisions(U_Net, train_no_shuffle, option=None)
# print('TopFormer的测试精度:'), test.get_precisions(Top_Former, train_no_shuffle, option=None)
# print('MANet的测试精度:', test.get_precisions(MA_Net, train_no_shuffle, option=None))
########################################################################################################################
