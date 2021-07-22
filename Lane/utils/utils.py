import smtplib
import torch
import yaml
import torch.nn as nn
import cv2
import numpy as np

from easydict import EasyDict as ed
from email.mime.text import MIMEText


def load_config(config_dir):
    return ed(yaml.load(open(config_dir), yaml.FullLoader))


def to_cuda(sample):
    for key in sample.keys():
        if type(sample[key]) == torch.Tensor:
            sample[key] = sample[key].cuda()
    return sample