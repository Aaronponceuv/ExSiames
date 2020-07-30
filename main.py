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

import wandb
wandb.init(project="tripletloss")

import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":

    training_dir="./Chatarra Conjunto 2/"
    training_csv="./Chatarra Conjunto 2/train_dif_simil.csv"
    val_csv = "./Chatarra Conjunto 2/val_dif_simil.csv"
    
    """
    train_an_pos_label = pd.read_csv("./test_an_pos_label.csv")
    train_dif = pd.read_csv("./test_dif.csv")
    #df_train_igual = pd.read_csv(train_igual)
    #val_an_pos["label"] = 1
    con = pd.concat([train_an_pos_label, train_dif])
    con.to_csv("test_dif_simil.csv", index=None)
    print(train_an_pos)
    """

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
                        shuffle=False,
                        num_workers=2,
                        batch_size=1)
    

    model = models.vgg16(pretrained=True)
    print(model)
    model.classifier[6] = nn.Linear(4096, 1)
    print(model)
    wandb.watch(model)
    net = SiamesNet(model)

    if torch.cuda.is_available():
        print('Cuda Disponible')
        net.cuda()
    
    criterion = ContrastiveLoss()
    
    optimizer = optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)

    for epoch in range(0,1000):
        train(train_dataloader, net, criterion, optimizer, epoch)
        validacion(val_dataloader, net, criterion, epoch)
        torch.save(model.state_dict(), "VGG16-Siames"+str(epoch)+".h5") 