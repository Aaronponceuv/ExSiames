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

from LogisticRegression import LogisticRegression
from AverageMeter import AverageMeter

from SiamesDataset import SiamesDataset
from SiamesNet import SiamesNet
import torchvision.models as models

from ContrastiveLoss import ContrastiveLoss

#import wandb
#wandb.init(project="tripletloss")

import numpy as np
import torch
torch.manual_seed(5)
np.random.seed(5)

torch.autograd.set_detect_anomaly(True) 

def train(train_dataloader,net, regresion, criterion,optimizer, epoch):
    registro = pd.DataFrame(columns=("batch_id",'x','sigmoid(x)','loss','pred'))
    counter = []
    loss_history = [] 
    iteration_number= 0
    net.train()
    
    losses = AverageMeter()
    accuracy = AverageMeter()
    for batch_idx, (img0, img1 , label) in enumerate(train_dataloader):
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()

        img0, img1  = Variable(img0), Variable(img1)
        output1,output2 = net(img0,img1)
        resta = (output1-output2).cuda()
        #Similutud cuadrada Chi
        #resta =  torch.pow(resta,2)/(output1+output2)
        resta =  torch.abs(torch.pow(resta,2)/(output1+output2))
        out_regresion = regresion(resta.cuda())
        loss = F.binary_cross_entropy_with_logits(out_regresion, label)
        #loss = criterion(out_regresion, label)
        registro.loc[batch_idx] = [batch_idx,resta.data[0],out_regresion.data[0],loss.item(),out_regresion.data[0]]
        registro.to_excel("./debug.xlsx")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = regresion.acc(out_regresion.round(),label)    
        accuracy.update(acc,img1.size(0))
        losses.update(loss.item(), img1.size(0))

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.4f} ({:.4f}) \t'.format(
                      epoch, batch_idx * len(img0), len(train_dataloader.dataset),
                      losses.val, losses.avg,accuracy.val,accuracy.avg))
    #wandb.log({
    #    "Train Accuracy": accuracy.avg,
    #    "Train Loss": losses.avg})
    return net

def validacion(test_dataloader,net, regresion,criterion, epoch):
    net.eval()
    accuracy = AverageMeter()
    losses = AverageMeter()
    with torch.no_grad():
        for batch_idx, (img0, img1 , label) in enumerate(test_dataloader): 
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            img0, img1  = Variable(img0), Variable(img1)
            output1,output2 = net(img0,img1)
            resta = (output1-output2).cuda()
            #Similutud cuadrada Chi
            resta =  torch.abs(torch.pow(resta,2)/(output1-output2))
            out_regresion = regresion(resta.cuda())
            acc = regresion.acc(out_regresion.round(),label)
            accuracy.update(acc,img0.size(0)) 
            loss = criterion(out_regresion, label)
            losses.update(loss,img0.size(0))
            
            print('Validation Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.4f} ({:.4f}) \t'.format(
                      epoch, batch_idx * len(img0), len(test_dataloader.dataset),
                      losses.val, losses.avg,accuracy.val,accuracy.avg))
       # wandb.log({"Val Accuracy": accuracy.avg,"Val Loss": losses.avg})


if __name__ == "__main__":

    training_dir="./Chatarra Conjunto 2/"
    training_csv="./Chatarra Conjunto 2/train_dif_simil.csv"
    val_csv = "./Chatarra Conjunto 2/val_dif_simil.csv"
    

    train_siamese_dataset = SiamesDataset(training_csv,
                                            training_dir,
                                            transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    
    train_dataloader = DataLoader(train_siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=32)

    val_siamese_dataset = SiamesDataset(val_csv,
                                            training_dir,
                                            transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    
    val_dataloader = DataLoader(val_siamese_dataset,
                        shuffle=True,
                        num_workers=2,
                        batch_size=32)
    

    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 128)
    print(model)
    net = SiamesNet(model)
    regresion = LogisticRegression(128,1)

    if torch.cuda.is_available():
        print('Cuda Disponible')
        net.cuda()
        regresion.cuda()
    
    #criterion = ContrastiveLoss()
    criterion = torch.nn.BCELoss()
    #criterion =torch.nn.CrossEntropyLoss()
    
    #optimizer = optim.RMSprop(regresion.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)
    optimizer = torch.optim.Adam(regresion.parameters(), lr=0.0001)
    #wandb.watch(regresion)

    for epoch in range(0,1000):
        train(train_dataloader, net,regresion, criterion, optimizer, epoch)
        #validacion(val_dataloader, net,regresion, criterion, epoch)
        #torch.save(regresion.state_dict(), "VGG16-Siames-Regresion-Chi.h5") 
