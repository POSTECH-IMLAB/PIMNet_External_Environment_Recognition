import cv2
import numpy as np
import glob

img_array = []

img = cv2.imread('cover.jpg')

for i in range(0,15):
    img_array.append(img)

for filename in sorted(glob.glob('./result3/*.jpg')):
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'MP4V'),15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
