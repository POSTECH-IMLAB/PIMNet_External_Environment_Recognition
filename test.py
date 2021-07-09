import numpy as np
import os
import cv2
from PIL import Image
from utils import box_iou, box_nms, change_box_order, meshgrid
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import preprocessing.transforms as transforms
from encoder import DataEncoder
from loss import *
from retinanet import RetinaNet
from preprocessing.datasets import VocLikeDataset
import matplotlib.pyplot as plt
import config as cfg
from tqdm import tqdm
import evaluate.anno_func
import json
from optparse import OptionParser


def load_model(backbone):
    print('loading model...')
    model= torch.load(os.path.join('ckpts', 'model',backbone+'_retinanet.pth'))
    net=RetinaNet(backbone=backbone,num_classes=len(cfg.classes))
    net=torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()
    cudnn.benchmark = True
    net.load_state_dict(model['net'])
    return net

def vis(img, boxes,boxes2, boxes3, labels, labels2, labels3, classes,color):
    img=img.copy()
  
    if boxes is not None:
        for box,label in zip(boxes,labels):
            margin = 10 
            cv2.rectangle(img, (max(0,int(box[0]))+margin+ 384,max(0,int(box[1]))+margin),(min(511,int(box[2]))+margin+ 384,min(511,int(box[3]))+margin),color,2)
            ss=cfg.classes[label-1]
            cv2.putText(img, ss, (int(box[0])+384,int(box[1]-10)), 0, 0.6, color, 2)

    print (boxes2)
    print (labels2)
    if boxes2 is not None:
        for box,label in zip(boxes2,labels2):
            margin = 10
            if box[0] > 896 - 768:
                cv2.rectangle(img, (max(0,int(box[0]))+margin + 768,max(0,int(box[1]))+margin),(min(511,int(box[2]))+margin + 768,min(511,int(box[3]))+margin),color,2)
                ss=cfg.classes[label-1]
                cv2.putText(img, ss, (int(box[0])+768,int(box[1]-10)), 0, 0.6, color, 2)

    if boxes3 is not None:
        for box,label in zip(boxes3,labels3):
            margin = 10
            if box[0] < 384 :
                cv2.rectangle(img, (max(0,int(box[0]))+margin,max(0,int(box[1]))+margin),(min(511,int(box[2]))+margin,min(511,int(box[3]))+margin),color,2)
                ss=cfg.classes[label-1]
                cv2.putText(img, ss, (int(box[0]),int(box[1]-10)), 0, 0.6, color, 2)
    return img



def eval_valid(net, valloader, anno_file,image_dir):
    net.eval()
    annos_pred={}
    annos_pred['imgs']={}
    annos=json.loads(open(anno_file).read())
    annos_target={}
    annos_target['imgs']={}
    
    for batch_idx, (inputs, loc_targets, cls_targets) in tqdm(enumerate(valloader)):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        loc_preds, cls_preds = net(inputs)
        for i in range(loc_preds.size()[0]):
            imgid=cfg.val_imageset_fn[batch_idx*batch_size+i].split('/')[-1][:-4]
            annos_target['imgs'][imgid]=annos['imgs'][imgid]
            boxes,labels,score=DataEncoder().decode(loc_preds[i], cls_preds[i], input_size=512)
            annos_pred['imgs'][imgid]={}
            rpath=os.path.join(image_dir,imgid+'.jpg')
            annos_pred['imgs'][imgid]['path']=rpath
            annos_pred['imgs'][imgid]['objects']=[]
            if boxes is None:
                continue
            for i,box in enumerate(boxes):
                bbox={}
                bbox['xmin']=box[0]
                bbox['xmax']=box[2]
                bbox['ymin']=box[1]
                bbox['ymax']=box[3]

            
                annos_pred['imgs'][imgid]['objects'].append({'score':100*float(score[i]),'bbox':bbox,'category':cfg.classes[labels[i]-1]})
    
    print('Test done, evaluating result...')
    
    
    with open(os.path.join(datadir,predict_dir),'w') as f:
        json_str=json.dumps(annos_pred)
        json.dump(annos_pred,f)
        f.close()
    with open(os.path.join(datadir,target_dir),'w') as f:
        json_str=json.dumps(annos_target)
        json.dump(annos_target,f)
        f.close()
    
def test_image(net, imgid_path,file_name):
    img=Image.open(os.path.join(imgid_path,file_name))
    area2 = (768,0, 1280, 512)
    area1 = (384,0, 896, 512)
    area3 = (0, 0, 512, 512)
    orig_img = img.copy()

    orig_img=np.asarray(orig_img)
    crop_img1 = img.crop(area1)
    crop_img2 = img.crop(area2)
    crop_img3 = img.crop(area3)
    #img=cv2.resize(np.lofimg,(512,512))
 
    
    img = crop_img1
    width, height=img.size
    #width = 512
    #height = 512
    #if width!=cfg.width or height!=cfg.height:
    #    img=cv2.resize(np.float32(img),(cfg.width, cfg.height))
    img=np.asarray(img)
    image = img.transpose((2, 0, 1))
    image=torch.from_numpy(image)
    if isinstance(image, torch.ByteTensor):
        image = image.float().div(255)
    for t, m, s in zip(image, cfg.mean, cfg.std):
        t.sub_(m).div_(s)
    net.eval()
    image=Variable(image.resize_(1,3,cfg.width,cfg.height))
    loc_pred, cls_pred=net(image)
    boxes,labels,score=DataEncoder().decode(loc_pred[0], cls_pred[0], input_size=(cfg.width,cfg.height))

    img2 = crop_img2
    width, height=img2.size
    #width = 512
    #height = 512
    #if width!=cfg.width or height!=cfg.height:
    #    img=cv2.resize(np.float32(img),(cfg.width, cfg.height))
    img2=np.asarray(img2)
    image2 = img2.transpose((2, 0, 1))
    image2=torch.from_numpy(image2)
    if isinstance(image2, torch.ByteTensor):
        image2 = image2.float().div(255)
    for t, m, s in zip(image2, cfg.mean, cfg.std):
        t.sub_(m).div_(s)
    net.eval()
    image2=Variable(image2.resize_(1,3,cfg.width,cfg.height))
    loc_pred, cls_pred=net(image2)
    boxes2,labels2,score=DataEncoder().decode(loc_pred[0], cls_pred[0], input_size=(cfg.width,cfg.height))

    img = crop_img3
    width, height=img.size
    #width = 512
    #height = 512
    #if width!=cfg.width or height!=cfg.height:
    #    img=cv2.resize(np.float32(img),(cfg.width, cfg.height))
    img=np.asarray(img)
    image = img.transpose((2, 0, 1))
    image=torch.from_numpy(image)
    if isinstance(image, torch.ByteTensor):
        image = image.float().div(255)
    for t, m, s in zip(image, cfg.mean, cfg.std):
        t.sub_(m).div_(s)
    net.eval()
    image=Variable(image.resize_(1,3,cfg.width,cfg.height))
    loc_pred, cls_pred=net(image)
    boxes3,labels3,score=DataEncoder().decode(loc_pred[0], cls_pred[0], input_size=(cfg.width,cfg.height))

 
    new_img=vis(orig_img, boxes, boxes2, boxes3, labels, labels2, labels3, cfg.classes, (0,255,0))
    return new_img
    
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-m', '--mode', dest='mode',default='demo',
		help='Operating mode, could be demo or valid, demo mode will provide visulization results for images in samples/')
    
    parser.add_option('--backbone','--backbone',dest='backbone',default='resnet152',
    help='Backbone pretrained model, could be resnet50, resnet101 or resnet152')
    
    options, args = parser.parse_args()
    mode = options.mode
    backbone=options.backbone
    if backbone not in ['resnet50', 'resnet101', 'resnet152']:
        assert ValueError('Invalid backbone: %s' % backbone)
    net=load_model(backbone)
        

    image_dir='samples'
    img_list=os.listdir(image_dir)
    for fname in img_list:
        print(fname)
        new_img=test_image(net, image_dir, fname)
        cv2.imwrite('./result/'+fname, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
        #plt.imshow(new_img)
        #plt.show()
        
    
