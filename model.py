import torch
import torch.nn as nn

from torch import Tensor
from torchvision import models

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class GeneratorHead(nn.Module):
    def __init__(self):
        super(GeneratorHead, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64,
                              kernel_size=9, stride=1, padding='same')
        self.PReLU = nn.PReLU(64)

    def forward(self, inp: Tensor):
        return self.PReLU(self.conv(inp))


class ResBlock(nn.Module):
    def __init__(self, dilation: bool = False):
        super(ResBlock, self).__init__()
        if dilation:
            self.conv1 = nn.Conv2d(in_channels=64, out_channels=64,
                                   kernel_size=3, stride=1, padding='same', dilation=2)
        else:
            self.conv1 = nn.Conv2d(in_channels=64, out_channels=64,
                                   kernel_size=3, stride=1, padding='same')

        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.PReLU1 = nn.PReLU(64)
        self.PReLU2 = nn.PReLU(64)

    def forward(self, inp: Tensor):
        X: Tensor = self.PReLU1(self.bn1(self.conv1(inp)))
        X = self.PReLU2(self.bn2(self.conv2(X)))
        X = self.bn3(self.conv3(X))
        return X.add(inp)  # Skip connections


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        self.attention=CBAM(256)
        # self.main = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=256,
        #               kernel_size=3, stride=1, padding=1),
            
        #     self.attention(),
        #     nn.PixelShuffle(upscale_factor=2),
        #     nn.PReLU(64)
        # )
        self.conv = nn.Conv2d(64,256, 3, 1, 1)
        self.ps = nn.PixelShuffle(upscale_factor=2)  # in_c * 4, H, W --> in_c, H*2, W*2
        self.act = nn.PReLU(num_parameters=64)

    def forward(self, inp: Tensor):
        inp=self.conv(inp)
        inp=self.attention(inp)
        inp=self.ps(inp)
        return self.act(inp)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.head = GeneratorHead()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.RRDB = nn.Sequential(
            *[ResBlock(dilation=True) if i % 2 == 0 else ResBlock() for i in range(16)])
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)
        self.tail = nn.Sequential(UpSample(), UpSample(), self.conv2)
        

    def forward(self, inp: Tensor) -> Tensor:
        preRRDB: Tensor = self.head(inp)
        X: Tensor = self.RRDB(preRRDB)
        X = self.bn(self.conv1(X))
        X = X.add(preRRDB)  # skip conn
        X = self.tail(X)
        return X


# ############# Discriminator ##############


class DiscriminatorHead(nn.Module):
    def __init__(self, concat: bool = True):
        super(DiscriminatorHead, self).__init__()
        self.concat = concat
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, inp: Tensor):
        
        return self.lrelu(self.conv(inp))


class DiscriminatorConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(DiscriminatorConvBlock, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=3,
                                            stride=stride),
                                  nn.BatchNorm2d(num_features=out_channels),
                                  nn.LeakyReLU(negative_slope=0.2))

    def forward(self, inp: Tensor):
        return self.main(inp)


class Flatten(nn.Module):
    def forward(self, x: Tensor):
        return x.view(x.size(0), -1)


class DiscriminatorTail(nn.Module):
    def __init__(self, patch: bool = True):
        super(DiscriminatorTail, self).__init__()
        if not patch:
            print(
                "\033[93m If you choose to have a non-patch discriminator, "
                "make sure the discriminator architecture is on accordance with your image size\033[0m"
            )
            self.main = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=12800, out_features=1024),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(in_features=1024, out_features=1),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            self.main = nn.Sequential(nn.Conv2d(in_channels=512,
                                                out_channels=1, kernel_size=3,
                                                stride=1, padding=1))

    def forward(self, inp: Tensor):
        return torch.sigmoid(self.main(inp))


class Discriminator(nn.Module):
    def __init__(self, patch: bool = True, concat: bool = False):
        super(Discriminator, self).__init__()
        self.head = DiscriminatorHead(concat=concat)
        self.body = nn.Sequential(DiscriminatorConvBlock(64, 64, 2),
                                  DiscriminatorConvBlock(64, 128, 1),
                                  DiscriminatorConvBlock(128, 128, 2),
                                  DiscriminatorConvBlock(128, 256, 1),
                                  DiscriminatorConvBlock(256, 256, 2),
                                  DiscriminatorConvBlock(256, 512, 1),
                                  DiscriminatorConvBlock(512, 512, 2),
                                  )
        self.tail = DiscriminatorTail(patch=True)

    def forward(self, inp: Tensor) -> Tensor:
        x: Tensor = self.head(inp)
        x = self.body(x)
        x = self.tail(x)
        return x

a=Discriminator()
b=Generator()
# ######## Perceptual Net ###########


# class PerceptionNet(nn.Module):
#     def __init__(self):
#         super(PerceptionNet, self).__init__()
#         modules = list(models.vgg19(pretrained=True).children())[0]
#         self.main = nn.Sequential(*modules)
#         for param in self.main.parameters():
#             param.requires_grad = False

#     def forward(self, inp):
#         return self.main(inp)