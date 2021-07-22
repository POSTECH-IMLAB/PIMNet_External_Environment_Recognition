import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

from .utils import *

class resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] = sample['image'].resize(self.size, Image.BILINEAR)
        if 'gt' in sample.keys():
            sample['gt'] = sample['gt'].resize(self.size, Image.NEAREST)
        if 'depth' in sample.keys():
            sample['depth'] = sample['depth'].resize(self.size, Image.NEAREST)

        return sample

class tonumpy:
    def __init__(self):
        pass

    def __call__(self, sample):
        for key in sample.keys():
            if key in ['image', 'gt', 'depth']:
                sample[key] = np.array(sample[key], dtype=np.float32)

        return sample

class normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] /= 255
            sample['image'] -= self.mean
            sample['image'] /= self.std

        if 'gt' in sample.keys():
            sample['gt'] /= 255

        if 'depth' in sample.keys():
            sample['depth'] /= 255

        return sample

class totensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] = sample['image'].transpose((2, 0, 1))
            sample['image'] = torch.from_numpy(sample['image']).float()
        
        if 'gt' in sample.keys():
            sample['gt'] = torch.from_numpy(sample['gt'])
            sample['gt'] = sample['gt'].unsqueeze(dim=0)

        if 'depth' in sample.keys():
            sample['depth'] = torch.from_numpy(sample['depth'])
            sample['depth'] = sample['depth'].unsqueeze(dim=0)

        return sample
