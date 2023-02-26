# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:15:52 2023

@author: Danish
"""

import torch.nn as nn
from torchvision.models import vgg19
import config
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import cv2
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
# phi_5,4 5th conv layer before maxpooling but after activation

# class VGGLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
#         self.loss = nn.MSELoss()

#         for param in self.vgg.parameters():
#             param.requires_grad = False

#     def forward(self, input, target):
#         vgg_input_features = self.vgg(input)
#         vgg_target_features = self.vgg(target)
#         return self.loss(vgg_input_features, vgg_target_features)

class custom_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1=nn.L1Loss()
        self.msssim_loss=1-MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        self.alpha=0.84
    def forward(self,X,Y):
        return torch.sum(self.alpha*self.msssim_loss(X,Y) +(1-self.alpha)*self.L1(X,Y))

import torch         
# a=torch.rand(1,3,400,400)
# b=torch.rand(1,3,400,400)


# c=custom_loss()
# print(c(a,b))
         