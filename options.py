"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/8/22 20:07 
"""

import argparse

class Options:
    def __init__(self):
        self.BATCH_SIZE: int = 5
        self.EPOCH: int = 1
        self.N_EPOCH: int = 1000
        self.LR: float = 1e-4
        self.BETA1: float = 0.9
        self.BETA2: float = 0.999

        self.TRAIN_IMG_ROOT: str = 'root/train'
        self.TRAIN_MASK_ROOT: str = 'root/mask'
        self.TRAIN_RESULT_ROOT: str = 'root/result/'
        self.SAVE_MODEL_ROOT: str = 'root/model/'

        self.IMG_SIZE: int = 256
        self.IN_C: int = 4
        self.OUT_C: int = 3
        self.PATCH_SIZE: int = 4
        self.EMBED_DIM: int = 64
        self.DEPTH: list = [1, 2, 3, 4]
        self.NUM_HEADS: list = [1, 2, 4, 8]

        self.ADV_LOSS_WEIGHT: float = 1.
        self.PER_LOSS_WEIGHT: float = 0.5
        self.STY_LOSS_WEIGHT: float = 1000.
        self.L1_LOSS_WEIGHT: float = 100.
        self.SOBEL_LOSS_WEIGHT: float = 80.

        self.SAMPLE_INTERVAL: int = 1000

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Description of your script')
        parser.add_argument('--BATCH_SIZE', type=int, default=self.BATCH_SIZE, help='Batch size for training')
        parser.add_argument('--EPOCH', type=int, default=self.EPOCH, help='Number of epochs')
        parser.add_argument('--N_EPOCH', type=int, default=self.N_EPOCH, help='Total number of epochs')
        parser.add_argument('--LR', type=float, default=self.LR, help='Learning rate')
        parser.add_argument('--BETA1', type=float, default=self.BETA1, help='Beta1 value for Adam optimizer')
        parser.add_argument('--BETA2', type=float, default=self.BETA2, help='Beta2 value for Adam optimizer')
        parser.add_argument('--TRAIN_IMG_ROOT', type=str, default=self.TRAIN_IMG_ROOT, help='Training image root directory')
        parser.add_argument('--TRAIN_MASK_ROOT', type=str, default=self.TRAIN_MASK_ROOT, help='Training mask root directory')
        parser.add_argument('--TRAIN_RESULT_ROOT', type=str, default=self.TRAIN_RESULT_ROOT, help='Training result root directory')
        parser.add_argument('--SAVE_MODEL_ROOT', type=str, default=self.SAVE_MODEL_ROOT, help='Directory to save models')
        parser.add_argument('--IMG_SIZE', type=int, default=self.IMG_SIZE, help='Image size')
        parser.add_argument('--IN_C', type=int, default=self.IN_C, help='Input channels')
        parser.add_argument('--OUT_C', type=int, default=self.OUT_C, help='Output channels')
        parser.add_argument('--PATCH_SIZE', type=int, default=self.PATCH_SIZE, help='Patch size')
        parser.add_argument('--EMBED_DIM', type=int, default=self.EMBED_DIM, help='Embedding dimension')
        parser.add_argument('--DEPTH', nargs='+', type=int, default=self.DEPTH, help='List of depths')
        parser.add_argument('--NUM_HEADS', nargs='+', type=int, default=self.NUM_HEADS, help='List of numbers of heads')
        parser.add_argument('--ADV_LOSS_WEIGHT', type=float, default=self.ADV_LOSS_WEIGHT, help='Weight of adversarial loss')
        parser.add_argument('--PER_LOSS_WEIGHT', type=float, default=self.PER_LOSS_WEIGHT, help='Weight of perceptual loss')
        parser.add_argument('--STY_LOSS_WEIGHT', type=float, default=self.STY_LOSS_WEIGHT, help='Weight of style loss')
        parser.add_argument('--L1_LOSS_WEIGHT', type=float, default=self.L1_LOSS_WEIGHT, help='Weight of L1 loss')
        parser.add_argument('--SOBEL_LOSS_WEIGHT', type=float, default=self.SOBEL_LOSS_WEIGHT, help='Weight of Sobel loss')
        parser.add_argument('--SAMPLE_INTERVAL', type=int, default=self.SAMPLE_INTERVAL, help='Interval for saving samples')

        args = parser.parse_args()
        return args

