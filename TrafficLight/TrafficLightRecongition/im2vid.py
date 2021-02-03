import cv2
import numpy as np
import os
img_path = 'D:/project/LISA/images/test/'
img_list = os.listdir(img_path)
img_list = sorted(img_list,key=lambda x: int(os.path.splitext(x)[0]))
img_array = []
for filename in img_list:
    # if i<3010:
    #     continue
    # if i==4000:
    #     break
    img = cv2.imread(img_path+filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    print(filename)

out = cv2.VideoWriter('D:/project/yolov5-master/TL.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()