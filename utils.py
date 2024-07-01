"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/8/22 20:02 
"""

from torchvision import transforms
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


def psnr_value(raw_tensor, gen_tensor, device):
    return PeakSignalNoiseRatio().to(device)(raw_tensor, gen_tensor)


def ssim_value(raw_tensor, gen_tensor, device):
    return StructuralSimilarityIndexMeasure().to(device)(raw_tensor, gen_tensor)


def mape_value(raw_tensor, gen_tensor, device):
    return MeanAbsolutePercentageError().to(device)(raw_tensor, gen_tensor)


def image_to_tensor(x):
    return transforms.ToTensor()(x)

def tensor_to_image(x):
    return transforms.ToPILImage()(x)



