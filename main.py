import torch
import torch.nn as nn

import Init_Dataset.Load as Load

image_root_dir = ['E:\\gzr\\SS_LoveDA\\Train\\Rural\\images_png',
                  'E:\\gzr\\SS_LoveDA\\Train\\Urban\\images_png']

mask_root_dir = ['E:\\gzr\\SS_LoveDA\\Train\\Rural\\masks_png',
                 'E:\\gzr\\SS_LoveDA\\Train\\Urban\\masks_png']

# 语义掩码
mask_unique = torch.tensor([0.0000, 0.0039, 0.0078, 0.0118, 0.0157, 0.0196, 0.0235, 0.0275])


