import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self,n_features,n_classes):
        super(LogisticRegression, self).__init__()
        self.layer1 = nn.Linear(n_features, n_classes)
 
    def forward(self, x):
        x = self.layer1(x)
        return F.sigmoid(x)

    def predict(self, x):
        # a function to predict the labels of a batch of inputs
        x = self.forward(x)
        return x

    def accuracy(self, x, y):
        # a function to calculate the accuracy of label prediction for a batch of inputs
        #   x: a batch of inputs
        #   y: the true labels associated with x
        prediction = self.predict(x)
        print(x.size())
        maxs, indices = torch.max(prediction, 1)
        print(torch.eq(indices.float(), y.float()))
        print(torch.eq(indices.float(), y.float()).float())
        print("sum",torch.sum(torch.eq(indices.float(), y.float()).float()))
        print(y.size())
        acc = torch.sum(torch.eq(indices.float(), y.float()).float())/y.size()[0]
        return acc.cpu()

    def acc(self,y_pred_tag,y_test):
        correct_results_sum = torch.eq(y_pred_tag,y_test).float().sum()
        acc = correct_results_sum/y_test.shape[0]
        acc = acc * 100
        return acc
