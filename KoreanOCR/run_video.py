import os
from re import L
import sys
import glob
import time
import torch

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def draw_ko(np_img, text, pos, fontsize=50):
    b, g, r, a = 255, 255, 255, 0
    fontpath = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    font = ImageFont.truetype(fontpath, fontsize)
    img_pil = Image.fromarray(np_img)
    draw = ImageDraw.Draw(img_pil)
    x, y = pos
    y -= (fontsize + 5)
    pos = (x, y)
    draw.text(pos, text, font=font, fill=(b, g, r, a))
    return np.asarray(img_pil)


def run(model, filename, vis_thres=0.3):
    savename = filename.replace('.MOV', '')
    cap = cv2.VideoCapture(filename)
    idx = 0

    while True:
        ret, image = cap.read()
        if image is None:
            break
        image = cv2.resize(image, (1024,604))
        
        start = time.time()
        result = model.ocr(image)
        end = time.time()
        
        for res in result:
            prob = res[2]
            if prob < vis_thres:
                continue
            box = [(round(b[0]), round(b[1])) for b in res[0]]
            image = cv2.rectangle(image, box[0], box[2], (0, 0, 255), 3)
            image = draw_ko(image, res[1] + f" ({prob:.2f})", tuple(res[0][0]))

        cv2.imwrite(f"{savename}_{idx:03d}.png", image)
        idx += 1
        print(end-start)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_dir = './checkpoints/last.ckpt'
    file_list = glob.glob(os.path.join("./video/", "IMG*.MOV"))
    reader_ko = torch.load(model_dir)

    for video in file_list:
        run(reader_ko, video, 0.2)
