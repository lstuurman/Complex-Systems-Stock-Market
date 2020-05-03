import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import pickle
import numpy as np
import glob
import sklearn.preprocessing as sk_prep

from train_eval import *

class  CNN_LSTM_predictor(nn.Module):
    def __init__(self,inshape,nfilters,kernelsize,lstmsize,hiddensize):
        super(CNN_LSTM_predictor,self).__init__()

        # shape parameters: 
        self.batch_size, self.seq = inshape
        self.n_filters = nfilters
        self.lstm_input = conf_len = self.seq - kernelsize + 1
        self.lstm_output = lstmsize
        # layers
        self.cnnPrice = nn.Conv1d(1,self.n_filters,kernelsize)
        self.cnnVolume = nn.Conv1d(1,self.n_filters,kernelsize)
        self.lstmPrice = nn.LSTM(conf_len,lstmsize)
        self.lstmVolume = nn.LSTM(conf_len,lstmsize)
        print(conf_len,lstmsize,nfilters)
        bilin_size = nfilters * conf_len
        self.Bilin = nn.Bilinear(bilin_size,bilin_size,hiddensize)
        self.output_layer = nn.Sequential(     
            nn.Dropout(p=0.5),  # explained later
            nn.Linear(hiddensize, 1)
            )
        # self.MP1 = nn.MaxPool1d(kernel_size = 2)      Maybe later  
        # self.MP2 = nn.MaxPool1d(kernel_size = 2)

    def forward(self,input):
        price,volume = input
        price = price.view(-1,1,self.seq)
        volume = volume.view(-1,1,self.seq)
        # keep track of batch size --> is smaller for last batch
        b_size = price.size()[0]
        print(b_size)

        ### Price : 
        # convolution 1
        out1 = self.cnnPrice(price)
        out1 = F.relu(out1)
        ### Volume :
        # convolution 1
        out2 = self.cnnVolume(volume)
        out2 = F.relu(out2)

        print(out1.size())
        ### Lstms
        price_Hns, price_Cns = ([],[])
        volume_Hns, volume_Cns = ([],[])
        # Loop over channels : 
        for ch in range(self.n_filters):
            # hidden init : 
            hiddenP = (torch.zeros(1,self.n_filters, self.lstm_input),torch.zeros(1, self.n_filters, self.lstm_input))
            hiddenV = (torch.zeros(1, self.n_filters, self.lstm_input),torch.zeros(1, self.n_filters, self.lstm_input))

            for i in range(1,self.lstm_input + 1):
                print(out1.size())
                print(out1[:,ch,:2].size())
                outp,hiddenP = self.lstmPrice(out1[:,ch,:i].view(self.batch_size,i,1)) # .view(1,self.n_filters,self.lstm_input)
                outv,hiddenV = self.lstmVolume(out2[:,ch,:i].view(self.batch_size,i,1))
                print(hiddenP[0].size())
                # print(hiddenP[1].size())
                # print(outp.size())
                price_Hns.append(hiddenP[0])
                price_Cns.append(hiddenP[1])
                volume_Hns.append(hiddenV[0])
                volume_Cns.append(hiddenV[1])
        
        #lstm_price = torch.tensor([torch.cat(lstm_price[i],lstm_price[i+1]) for i in range(len(lstm_price) -1)])
        price_Hns = torch.stack(price_Hns,dim = 0)
        #price_out = torch.cat(torch.stack(outputs, dim=0) ,torch.tensor(price_Cns))
        print(price_Hns.size())
        exit()
            
        
        #lstm_output_Price , _ = self.lstmPrice(out1)


        ### Lstm 1
        lstm_output_Volume, _ = self.lstmVolume(out2)

        # flatten channels : 
        lstm_output_Price = lstm_output_Price.view(b_size,-1)
        lstm_output_Volume = lstm_output_Volume.view(b_size,-1)
        print(lstm_output_Price.size())
        output = self.Bilin(lstm_output_Price,lstm_output_Volume)

        output = self.output_layer(output)

        return output

    def prepare_minibatch(self,data_file):
        # data is text file containing volume,price for one stock
        data = np.loadtxt(data_file)
        price,volume = data.T
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

        inseq = self.seq
        seq = self.seq + 1
        
        batches = []
        targets = []

        for i in range(len(data) - seq):
            vol_slice = sk_prep.minmax_scale(volume[i:i+inseq].reshape(-1,1),copy = True)
            pr_slice = sk_prep.minmax_scale(price[i:i+seq].reshape(-1,1),copy = True)
            batches.append((pr_slice[:-1],vol_slice))
            targets.append(pr_slice[-1].reshape(-1))
        
        batches = torch.Tensor(batches).to(device)
        targets = torch.Tensor(targets).to(device)

        minibatches = torch.split(batches,50)
        minitargets = torch.split(targets,50)
        
        return minibatches,minitargets







if __name__ == "__main__":
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = CNN_LSTM_predictor((50,50),20,10,1,50)
    model = model.to(device)
    #dfile = '../stock_data/NASDAQ/A'
    #model.prepare_minibatch(dfile)
    #evaluate(model,dfile)
    train(model,'Losses50.pkl','ACC50.pkl')
