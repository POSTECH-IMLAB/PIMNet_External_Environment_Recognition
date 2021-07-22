import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from .backbones.Res2Net_v1b import res2net50_v1b_26w_4s

class Model(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channels=64, pretrained=True):
        super(Model, self).__init__()
        self.backbone = res2net50_v1b_26w_4s(pretrained=pretrained)

        self.context1 = encoder(64, channels)
        self.context2 = encoder(256, channels)
        self.context3 = encoder(512, channels)
        self.context4 = encoder(1024, channels)
        self.context5 = encoder(2048, channels)

        self.decoder = decoder(channels)

        self.attention =  context(channels    , channels, lmap_in=True)
        self.attention1 = context(channels * 2, channels, lmap_in=True)
        self.attention2 = context(channels * 2, channels)

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        self.pyr = pyr(7, 1)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x1 = self.backbone.relu(x)
        x2 = self.backbone.maxpool(x1)

        x2 = self.backbone.layer1(x2)
        x3 = self.backbone.layer2(x2)
        x4 = self.backbone.layer3(x3)
        x5 = self.backbone.layer4(x4)

        x1 = self.context1(x1)
        x2 = self.context2(x2)
        x3 = self.context3(x3)
        x4 = self.context4(x4)
        x5 = self.context5(x5)

        f3, d3 = self.decoder(x5, x4, x3)

        f2, p2 = self.attention2(torch.cat([x2, self.ret(f3, x2)], dim=1), d3.detach()) 
        d2 = self.pyr.rec(d3.detach(), p2)

        f1, p1 = self.attention1(torch.cat([x1, self.ret(f2, x1)], dim=1), d2.detach(), p2.detach())
        d1 = self.pyr.rec(d2.detach(), p1)

        _, p = self.attention(self.res(f1, (H, W)), d1.detach(), p1.detach())
        d = self.pyr.rec(d1.detach(), p)
        d =  self.res(d, (H, W))
        return d