import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        #print("val ", self.val)
        self.sum += val * n
        #print("sum", self.sum)
        #print("tam batch",n) 
        self.count += n # Total de Batch procesados
        #print("count", self.count)
        self.avg = self.sum / self.count
        #print("avg ", self.avg)