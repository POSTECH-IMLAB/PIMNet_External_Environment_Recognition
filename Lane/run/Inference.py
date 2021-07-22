import torch
import os
import argparse
import tqdm
import sys
import cv2

import torch.nn.functional as F
import numpy as np

from PIL import Image

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.utils import *
from utils.dataloader import *
from utils.custom_transforms import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/HighwayLane.yaml')
    parser.add_argument('--source', type=str)
    return parser.parse_args()


def inference(opt, args):
    model = Model(channels=64, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(
        opt.Test.Checkpoint.checkpoint_dir, 'latest.pth')), strict=True)
    model.cuda()
    model.eval()
    
    transform = eval(opt.Test.Dataset.type).get_transform(
        opt.Test.Dataset.transform_list)

    source_dir = args.source
    source_list = os.listdir(args.source)
    source_list.sort()

    save_dir = os.path.join('results', args.source.split(os.sep)[-1])
    os.makedirs(save_dir, exist_ok=True)

    sources = tqdm.tqdm(enumerate(source_list), desc='Inference', total=len(
        source_list), position=0, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        
    for i, source in sources:
        img = Image.open(os.path.join(source_dir, source)).convert('RGB')
        sample = {'image': img}
        sample = transform(sample)
        x = sample['image'].unsqueeze(0)
        with torch.no_grad():
            out = model(x.cuda())
        out = F.interpolate(
            out, img.size[::-1], mode='bilinear', align_corners=True)
        out = out.data.cpu()
        out = torch.sigmoid(out)
        out = out.numpy().squeeze()
        out = (out - out.min()) / \
            (out.max() - out.min() + 1e-8)
        out = (out > .5)
        
        img = np.array(img)
        img[out != 0, :] = [0, 255, 0]
        Image.fromarray(img).save(os.path.join(
            save_dir, os.path.splitext(source)[0] + '.png'))


if __name__ == "__main__":
    args = _args()
    opt = load_config(args.config)
    inference(opt, args)
