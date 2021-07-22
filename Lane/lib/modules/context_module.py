import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *

class simple_context(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(simple_context, self).__init__()
        self.branch0 = conv(in_channel, out_channel, 1)
        self.branch1 = conv(in_channel, out_channel, 3, dilation=3)
        self.branch2 = conv(in_channel, out_channel, 3, dilation=5)
        self.branch3 = conv(in_channel, out_channel, 3, dilation=7)

        self.conv_cat = conv(4 * out_channel, out_channel, 3, relu=True)
        self.conv_res = conv(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat([x0, x1, x2, x3], dim=1)
        x_cat = self.conv_cat(x_cat)

        x_cat = x_cat + self.conv_res(x)
        return x_cat

class kernel(nn.Module):
    def __init__(self, in_channel, out_channel, receptive_size=3):
        super(kernel, self).__init__()
        self.conv0 = conv(in_channel, out_channel, 1)
        self.conv1 = conv(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = conv(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = conv(out_channel, out_channel, 3, dilation=receptive_size)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(x)

        x = self.conv3(Hx + Wx)
        return x

class encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(encoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = conv(in_channel, out_channel, 1)
        self.branch1 = kernel(in_channel, out_channel, 3)
        self.branch2 = kernel(in_channel, out_channel, 5)
        self.branch3 = kernel(in_channel, out_channel, 7)

        self.conv_cat = conv(4 * out_channel, out_channel, 3)
        self.conv_res = conv(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x