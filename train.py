from torch.autograd import Variable

from AverageMeter import AverageMeter
import torch
import wandb
def accuracy_(output1,output2,label):
    correct = 0
    res=torch.abs(output1.cuda() - output2.cuda())
    print(label)

    label=label[0].tolist()
    label=int(label[0])
        #print(torch.max(res,1))
    result=torch.max(res,1)[1].data[0].tolist()
    print(torch.max(res,1)[1].data[0])
    if label == result:
        correct=correct+1
    print(correct)

def train(train_dataloader,net,criterion,optimizer, epoch):
    counter = []
    loss_history = [] 
    iteration_number= 0
    net.train()
    
    losses = AverageMeter()
    for batch_idx, (img0, img1 , label) in enumerate(train_dataloader):
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()

        img0, img1  = Variable(img0), Variable(img1)
        optimizer.zero_grad()
        output1,output2 = net(img0,img1)
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()

        losses.update(loss_contrastive.item(), img1.size(0))
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'.format(
                      epoch, batch_idx * len(img0), len(train_dataloader.dataset),
                      losses.val, losses.avg))
    wandb.log({"Train Loss": losses.avg})    
    return net

def validacion(test_dataloader,net, criterion, epoch):
    accuracy=0
    counter=0
    correct=0
    net.eval()
    incorrect = 0
    for batch_idx, (img0, img1 , label) in enumerate(test_dataloader): 
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        img0, img1  = Variable(img0), Variable(img1)
        output1,output2 = net(img0,img1)

        res=torch.abs(output1.cuda() - output2.cuda())
        label=label[0].tolist()
        label=int(label[0])
        result=torch.max(res,1)[1].data[0].tolist()
        if label == result:
            correct=correct+1
           #print("Validation Epoch: {} Accuracy:{}%".format(epoch,(correct/(batch_idx+1))*100))
    accuracy=(correct/len(test_dataloader))*100
    print("Validation Epoch: {} Accuracy:{}%".format(epoch,accuracy))
    wandb.log({"Test Accuracy": accuracy})
