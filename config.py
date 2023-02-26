# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:12:53 2023

@author: Danish
"""

import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "gen1.pth.tar"
CHECKPOINT_DISC = "disc1.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 1
NUM_WORKERS = 4
HIGH_RES_WIDTH = 640
HIGH_RES_HEIGHT=480
LOW_RES_WIDTH = HIGH_RES_WIDTH // 4
LOW_RES_HEIGHT=HIGH_RES_HEIGHT//4
IMG_CHANNELS = 3

highres_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES_WIDTH, height=LOW_RES_HEIGHT, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES_WIDTH, height=HIGH_RES_HEIGHT),
        A.HorizontalFlip(p=0.5),
        # A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

