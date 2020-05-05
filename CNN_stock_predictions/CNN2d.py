import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import pickle
import numpy as np
import glob
import sklearn.preprocessing as sk_prep
from random import shuffle

from train_eval import *

class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D,self).__init__()

        #layers : 
        self.cnn1 = nn.Conv2d(1,50,(2,10)).double()
        self.linear = nn.Linear(2050,1)

    def forward(self,x): 
        x1,x2 = x
        t = torch.cat((x1.view(50,1,50,1),x2.view(50,1,50,1)),3)
        t = F.relu(self.cnn1(t.double()))
        t.squeeze_(2)
        return self.linear(t.float())

    def prepare_minibatch(self,data_file):
        # data is text file containing volume,price for one stock
        data = np.loadtxt(data_file)
        price,volume = data.T
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        batches = []
        targets = []

        for i in range(len(data) - 21):
            vol_slice = sk_prep.minmax_scale(volume[i:i+20].reshape(-1,1),copy = True)
            pr_slice = sk_prep.minmax_scale(price[i:i+21].reshape(-1,1),copy = True)
            batches.append((pr_slice[:-1],vol_slice))
            targets.append(pr_slice[-1].reshape(-1))
        
        batches = torch.Tensor(batches).to(device)
        targets = torch.Tensor(targets).to(device)

        minibatches = torch.split(batches,50)
        minitargets = torch.split(targets,50)
        
        return minibatches,minitargets
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = CNN2D()
    model = model.to(device)
    #dfile = '../stock_data/NASDAQ/A'
    #model.prepare_minibatch(dfile)
    #evaluate(model,dfile)
    train(model)