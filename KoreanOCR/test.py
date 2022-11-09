
from models.mobilenetv2 import *
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from dataset import get
import sys
import os
sys.path.append(os.path.expanduser('~/pytorch-quant/utee'))
import quantization

def bn2conv(model):
    r""" conv layer must be arranged before bn layer !!!"""
    if isinstance(model,nn.Sequential):
        ikv = enumerate(model._modules.items())
        for i,(k,v) in ikv:
            if isinstance(v,nn.Conv2d):
                key,bn = next(ikv)[1]
                if isinstance(bn, nn.BatchNorm2d):
                    if bn.affine:
                        a = bn.weight / torch.sqrt(bn.running_var+bn.eps)
                        b = - bn.weight * bn.running_mean / torch.sqrt(bn.running_var+bn.eps) + bn.bias
                    else:
                        a = 1.0 / torch.sqrt(bn.running_var+bn.eps)
                        b = - bn.running_mean / torch.sqrt(bn.running_var+bn.eps)
                    v.weight = Parameter( v.weight * a.reshape(v.out_channels,1,1,1) )
                    v.bias   = Parameter(b)
                    model._modules[key] = nn.Sequential()
            else:
                bn2conv(v)
        
    else:
        for k,v in model._modules.items():
            bn2conv(v)

def eval_model(model, ds, n_sample=None, ngpu=1):
    import tqdm
    import torch
    from torch import nn
    device = torch.device("cuda")
    correct1, correct5 = 0, 0
    n_passed = 0
    model = model.eval()
    #model = torch.nn.DataParallel(model, device_ids=range(ngpu)).cuda()
    model = model.to(device)
    n_sample = len(ds) if n_sample is None else n_sample
    with torch.no_grad():
        for idx, (input, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
            n_passed += len(input) # n_passed is batch_size
            
            input =  torch.FloatTensor(input).to(device)
            #print("input:\n{}".format(input))
            #input = quant.linear_quantize(input, sf=6, bits=8)
            indx_target = torch.LongTensor(target)
            
            output = model(input)
            
            batch_size = output.size(0)
            idx_pred = output.data.sort(1, descending=True)[1]
            idx_gt1 = indx_target.expand(1, batch_size).transpose_(0, 1)
            idx_gt5 = idx_gt1.expand(batch_size, 5)

            correct1 += idx_pred[:, :1].cpu().eq(idx_gt1).sum().item()
            correct5 += idx_pred[:, :5].cpu().eq(idx_gt5).sum().item()
            
            if idx >= n_sample - 1:
                break
    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5

def eval(net):
    data_root="~/dataset/"
    val_loader = get(
                batch_size=256, 
                data_root=data_root, 
                train=False, 
                val=True, 
                shuffle=True)
    acc1, acc5 = eval_model(net,val_loader)
    print("acc1:{}, acc5:{}".format(acc1,acc5))

net =  mobilenetv2()
quantization.bn2conv(net)
state_dict = net.state_dict()
for k,v in state_dict.items():
    print(k)
    v, S, Z = quantization.scale_quantize(v,8)
    #print("quantized v:{}, S:{}, Z:{}".format(v,S,Z))
    print("S:{}, Z:{}".format(S,Z))