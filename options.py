"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2023/8/22 20:07 
"""


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
        self.STY_LOSS_WEIGHT: float = 10000.
        self.L1_LOSS_WEIGHT: float = 100.
        self.SOBEL_LOSS_WEIGHT: float = 80.

        self.SAMPLE_INTERVAL: int = 1000

