import torchvision
import torch.utils.data as utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import time
import copy
from torch.optim import lr_scheduler
import os
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd 

from SiamesDataset import SiamesDataset
from SiamesNet import SiamesNet
import torchvision.models as models

from ContrastiveLoss import ContrastiveLoss
from train import *


import numpy as np
import torch

def exactitud(vp,vn,fn,fp):
    exac = (vp+vn)/(vp+vn+fn+fp)
    return exac

def taza_de_error(vp,vn,fn,fp):
    error = (fp+fn)/(vp+vn+fn+fp)
    return error

def sensibilidad(vp,vn,fn,fp):
    sensi = vp/(vp+fn)
    return sensi

def especificidad(vp,vn,fn,fp):
    especifi = vn/(vn+fp)
    return especifi

def precision(vp,vn,fn,fp):
    prec= vp/(vp+p)
    return prec

def f1_score(vp,vn,fn,fp):
    recall = sensibilidad(vp,vn,fn,fp)
    prec = precision(vp,vn,fn,fp)
    f1 = 2*(Recall * prec)/(Recall + prec)

def validacion(test_dataloader,net, criterion, epoch=None):
    net.eval()
    vp = 0
    fn = 0
    vn = 0
    fp = 0
    for batch_idx, (img0, img1 , label) in enumerate(test_dataloader): 
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        img0, img1  = Variable(img0), Variable(img1)
        output1,output2 = net(img0,img1)

        res=torch.abs(output1.cuda() - output2.cuda())
        label=label[0].tolist()
        label=int(label[0])
        result=torch.max(res,1)[1].data[0].tolist()
        print("label: {} result:{}".format(label,result))
        if(label == 1):
            if(label == result):
                vp +=1
            else:
                fn +=1 
        if(label == 0):
            if(label == result):
                vn +=1
            else:
                fp +=1
        print("VP: {} \n VN: {} \n FN: {} \n FP: {}".format(vp,vn,fn,fp))
    accuracy = exactitud(vp,vn,fn,fp)
    precisi= precision(vp,vn,fn,fp)  
    sensibilidad = sensibilidad(vp,vn,fn,fp)
    especificidad = especificidad(vp,vn,fn,fp)
    f1_score = f1_score(vp,vn,fn,fp)
    print("Resultados:\nVP: {} \n VN: {} \n FN: {} \n FP: {}".format(vp,vn,fn,fp))
    print("Validation Accuracy:{}% \n precision: {}% \n sensibilidad: {}% \n Especificidad: {}% \n f1_score: {}%".format(accuracy,precisi,sensibilidad,especificidad,f1_score))



if __name__ == "__main__":
    training_dir="./Chatarra Conjunto 2/"
    test_csv = "./Chatarra Conjunto 2/test_dif_simil.csv"
    

    test_siamese_dataset = SiamesDataset(test_csv,
                                            training_dir,
                                            transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    
    test_dataloader = DataLoader(test_siamese_dataset,
                        shuffle=False,
                        num_workers=2,
                        batch_size=1)
    

    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 1)

    model.load_state_dict(torch.load("./h5/VGG16-Siames31.h5"))
    net = SiamesNet(model)

    if torch.cuda.is_available():
        print('Cuda Disponible')
        net.cuda()
    
    criterion = ContrastiveLoss()
    
    optimizer = optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)
    net.eval()
    validacion(test_dataloader, net, criterion)