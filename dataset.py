"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/8/16 22:43 
"""


import cv2
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SatelliteDateset(Dataset):
    def __init__(self, image_root, mask_root, image_type='image'):
        super(SatelliteDateset, self).__init__()

        self.image_files = sorted(glob.glob(image_root + '/*.*'))
        self.mask_files = sorted(glob.glob(mask_root + '/*.*'))

        self.number_image = len(self.image_files)
        self.number_mask = len(self.mask_files)
        self.image_type = image_type

    def __getitem__(self, index):
        if self.image_type == 'tensor':
            img_tensor = torch.load(self.image_files[index % self.number_image])
        elif self.image_type == 'image':
            img_tensor = self.image_to_tensor()(Image.open(self.image_files[index % self.number_image]))
        else:
            raise TypeError("The input can only be 'tensor' or 'image'.")
        img_gray = cv2.imread(self.image_files[index % self.number_image], cv2.IMREAD_GRAYSCALE)
        mask_tensor = self.image_to_tensor()(Image.open(self.mask_files[index % self.number_mask]))
        mask_tensor = (mask_tensor - torch.min(mask_tensor)) / (torch.max(mask_tensor) - torch.min(mask_tensor))

        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_z = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)

        img_sobel = torch.tensor(sobel_x ** 2 + sobel_y ** 2 + sobel_z ** 2, dtype=torch.float32).unsqueeze(0)
        sobel_mask = torch.abs(torch.sqrt(img_sobel * (1 - mask_tensor)))
        sobel_mask = (sobel_mask - torch.min(sobel_mask)) / (torch.max(sobel_mask) - torch.min(sobel_mask))

        sobel_tensor = np.abs(np.sqrt(img_sobel))
        sobel_tensor = (sobel_tensor - torch.min(sobel_tensor)) / (torch.max(sobel_tensor) - torch.min(sobel_tensor))

        return img_tensor, mask_tensor, sobel_mask, sobel_tensor

    def __len__(self):
        return self.number_image

    def image_to_tensor(self):
        return transforms.ToTensor()

    def tensor_to_image(self):
        return transforms.ToPILImage()