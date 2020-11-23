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



from train_eval2 import *

class  simple_CNN(nn.Module):
    def __init__(self,inshape,nfilters):
        super(simple_CNN,self).__init__()

        # shape parameters: 
        self.batch_size, self.seq = inshape
        self.n_filters = nfilters
        # layers
        # first convolution : 
        self.cnn1 = nn.Conv1d(1,nfilters,10)
        self.mp1 = nn.MaxPool1d(2)
        self.cnn2 = nn.Conv1d(50,100,5)
        self.mp2 = nn.MaxPool1d(2)
        self.fc = nn.Linear(100*20,1)

    def forward(self,x):
        #print('input shape : ',x.size())
        x = F.relu(self.cnn1(x.view(-1,1,100)))
        x = self.mp1(x)
        x = F.relu(self.cnn2(x))
        x = self.mp2(x)
        # flatten :
        output = self.fc(x.view(-1,100*20))
        return output


    def prepare_minibatch(self,data_file):
        # data is text file containing volume,price for one stock
        #print(data_file)
        # data = np.loadtxt(data_file) OLD DATA FORMAT
        # price,volume = data.T
        # NEW DATA FORMAT 09-10
        data = pd.read_csv(data_file,index_col = 0)
        data = data.to_numpy()
        price,volume = data.T
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        inseq = self.seq
        seq = self.seq + 1
        
        batches = []
        targets = []

        for i in range(len(data) - seq):
            vol_slice = sk_prep.minmax_scale(volume[i:i+inseq].reshape(-1,1),copy = True)
            pr_slice = sk_prep.minmax_scale(price[i:i+seq].reshape(-1,1),copy = True)
            concat = np.concatenate((pr_slice[:-1],vol_slice),axis = 0)
            #print(concat.shape)
            # exit()
            batches.append(concat)
            targets.append(pr_slice[-1].reshape(-1))
        
        batches = torch.Tensor(batches).to(device)
        targets = torch.Tensor(targets).to(device)

        minibatches = torch.split(batches,50)
        minitargets = torch.split(targets,50)
        
        return minibatches,minitargets







if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = simple_CNN((50,50),50)
    model.to(device)
    # batches,targets = model.prepare_minibatch('../stock_data/NASDAQ/A')
    # model(batches[0])
    train(model,'Smpl_CNN_loss.txt','Smpl_CNN_acc.txt')
