#encoding=utf-8

import os
import numpy as np

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms

from PIL import Image

img_to_tensor = transforms.ToTensor()

def make_model():
    resmodel=models.resnet152(pretrained=True)
    resmodel.cuda()
    return resmodel


def inference(resmodel,imgpath):
    resmodel.eval()
    
    img=Image.open(imgpath)
    img=img.resize((224,224))
    tensor=img_to_tensor(img)
    
    tensor=tensor.resize_(1,3,224,224)
    tensor=tensor.cuda()
            
    result=resmodel(Variable(tensor))
    result_npy=result.data.cpu().numpy()
    max_index=np.argmax(result_npy[0])
    
    return max_index
    

def extract_feature(resmodel,imgpath):
    resmodel.fc=torch.nn.LeakyReLU(0.1)
    resmodel.eval()
    
    img=Image.open(imgpath)
    img=img.resize((224,224))
    tensor=img_to_tensor(img)
    
    tensor=tensor.resize_(1,3,224,224)
    tensor=tensor.cuda()
            
    result=resmodel(Variable(tensor))
    result_npy=result.data.cpu().numpy()
    
    return result_npy[0]
    
if __name__=="__main__":
    model=make_model()
    imgpath='cat.jpg'
    print (inference(model,imgpath))
    print (len(extract_feature(model, imgpath)))

    model=make_model()
    imgpath='cat2.jpg'
    print (inference(model,imgpath))
    print (len(extract_feature(model, imgpath)))

    model=make_model()
    imgpath='dog.jpeg'
    print (inference(model,imgpath))
    print (len(extract_feature(model, imgpath)))

    

