"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/8/17 11:45 
"""

import torch
import glob
from PIL import Image
from torchvision import transforms


class CropSatelliteImage:
    def __init__(self, input_path, output_path):
        self.input_path = sorted(glob.glob(input_path + '/*.*'))
        self.output_path = output_path

    def rotation(self, img_list, degree=10, crop=5120, channels=3):
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomRotation(degrees=(degree, degree)),
                                    transforms.CenterCrop(crop)])
        img_list = img_list[:channels]
        tensor_cat = torch.cat([trans(Image.open(i)) for i in img_list], dim=0)
        return tensor_cat

    def tensor_to_image(self):
        return transforms.ToPILImage()

    def cut(self, chunk_number=20):

        img_list = self.input_path
        img_list[0], img_list[2] = img_list[2], img_list[0]
        cat_tensor = self.rotation(img_list=img_list)
        raw_img = self.tensor_to_image()(cat_tensor)

        tensor_chunk1 = torch.chunk(cat_tensor, chunks=chunk_number, dim=1)
        for j, chunk in enumerate(tensor_chunk1):
            tensor_chunk2 = torch.chunk(chunk, chunks=chunk_number, dim=2)
            for k, chunk_complete in enumerate(tensor_chunk2):
                pic = self.tensor_to_image()(chunk_complete.clone())
                pic.save(self.output_path + '/' + str(j + 1) + '_' + str(k + 1) + '.png')
