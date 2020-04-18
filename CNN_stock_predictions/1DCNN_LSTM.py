import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np
import glob
import sklearn.preprocessing as sk_prep

class  CNN_LSTM_predictor(nn.Module):
    def __init__(self,inshape,kernelsize,hiddensize):
        super(CNN_LSTM_predictor,self).__init__()

        # shape parameters: 
        self.batch_size, self.n_filters = inshape
        self.lstm_input = seq_len = 20 - kernelsize + 1
        
        # layers
        self.cnnPrice = nn.Conv1d(1,self.n_filters,kernelsize)
        self.cnnVolume = nn.Conv1d(1,self.n_filters,kernelsize)
        self.lstmPrice = nn.LSTM(self.lstm_input,self.lstm_input)
        self.lstmVolume = nn.LSTM(self.lstm_input,self.lstm_input)
        bilin_size = seq_len * self.n_filters
        self.Bilin = nn.Bilinear(bilin_size,bilin_size,36)
        self.output_layer = nn.Sequential(     
            nn.Dropout(p=0.5),  # explained later
            nn.Linear(36, 1)
            )
        # self.MP1 = nn.MaxPool1d(kernel_size = 2)      Maybe later  
        # self.MP2 = nn.MaxPool1d(kernel_size = 2)

    def forward(self,input):
        price,volume = input
        price = price.view(-1,1,20)
        volume = volume.view(-1,1,20)
        # keep track of batch size --> is smaller for last batch
        b_size = price.size()[0]

        ### Price : 
        # convolution 1
        out1 = self.cnnPrice(price)
        out1 = F.relu(out1)
        ### Lstm 1
        lstm_output_Price , _ = self.lstmPrice(out1)

        ### Volume :
        # convolution 1
        out2 = self.cnnVolume(volume)
        out2 = F.relu(out2)
        ### Lstm 1
        lstm_output_Volume, _ = self.lstmVolume(out2)

        # flatten channels : 
        lstm_output_Price = lstm_output_Price.view(b_size,-1)
        lstm_output_Volume = lstm_output_Volume.view(b_size,-1)

        output = self.Bilin(lstm_output_Price,lstm_output_Volume)

        output = self.output_layer(output)

        return output

    def prepare_minibatch(self,data_file):
        # data is text file containing volume,price for one stock
        data = np.loadtxt(data_file)
        price,volume = data.T
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        # normalize : 
        volume = sk_prep.minmax_scale(volume.reshape((-1,1)),copy = True)
        price = sk_prep.minmax_scale(volume.reshape((-1,1)), copy = True)
        
        batches = []
        targets = []

        for i in range(len(data) - 21):
            vol_slice = volume[i:i+20]
            pr_slice = price[i:i+20]
            batches.append((pr_slice,vol_slice))
            targets.append(price[i+20])
        
        batches = torch.Tensor(batches).to(device)
        targets = torch.Tensor(targets).to(device)

        minibatches = torch.split(batches,50)
        minitargets = torch.split(targets,50)

        return minibatches,minitargets

def evaluate(model,datafile):
    # stats to compute : 
    deviations = []
    corrects = 0
    corr_bulls = 0  # true positives
    corr_bears = 0  # true negatives
    false_bulls = 0
    false_bears = 0
    count = 0

    model.eval()

    batches,targets = model.prepare_minibatch(datafile)

    for b,batch in enumerate(batches):
        price,volume = torch.split(batch,1,dim = 1)
        target = targets[b]
        predictions = model((price,volume))

        for y_i,seq in enumerate(batch):
            seq = seq[0].numpy().flatten()
            t = target[y_i].numpy()[0]
            pred = predictions[y_i].detach().numpy()[0]
            # Relative error
            high = max(seq)
            low = min(seq)
            err = t - pred
            perc_dev = abs(err)/abs(high - low)
            deviations.append(perc_dev)

            # bullish :
            if t > seq[-1]:
                if pred > seq[-1]:
                    corr_bulls += 1
                    corrects += 1
                    count += 1
                elif pred < seq[-1]:
                    false_bears += 1
                    count += 1
            # bearish :
            elif t < seq[-1]:
                if pred < seq[-1]:
                    corr_bears += 1
                    corrects += 1
                    count += 1
                elif pred > seq[-1]:
                    false_bulls += 1
                    count += 1
    print('Average percentage error :',np.average(deviations))        
    print('Correct Bulls : ',corr_bulls, ' | Correct Bears :',corr_bears)
    print('False   Bulls : ',false_bulls, ' | False   Bears : ',false_bears)

    return np.average(deviations), corr_bulls,corr_bears,false_bulls,false_bears



def train(model):
    # devide data : 
    files = glob.glob('../stock_data/NASDAQ/*')
    train = files[:440]
    test = files[440:450]
    #evl = files[450:]

    # some usefull measures
    training_iters = 1000
    train_loss = 0.
    criterion = nn.MSELoss() # loss function
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    best_eval = 1000.
    best_iter = 0
    n_evals = training_iters/50
    losses = []
    eval_data = []

    #scheduler = StepLR(optimizer,step_size = 100,gamma = 0.1)

    for i in range(training_iters):
        for t_file in train:

            model.train()
            
            minibatches,targets = model.prepare_minibatch(t_file)

            for n,batch in enumerate(minibatches):
                price,volume = volume,price = torch.split(batch,1,dim = 1)
                target = targets[i]

                # forward
                outputs = model((price,volume))

                loss = criterion(outputs,target)
                train_loss += loss.item()
                losses.append(train_loss)

                # backward : 
                model.zero_grad()
                loss.backward()
                optimizer.step()

        # print some info : 
        if i % 20 == 0:
            print('Training loss : ',train_loss)
            train_loss = 0.

        # evaluate : 
        if i % n_evals == 0:
            print(i)
            devs,cbull,cbear,fbull,fbear = evaluate(model,test[int(i/n_evals)])
            eval_data.append([devs,cbull,cbear,fbull,fbear])

            if devs < best_eval:
                best_eval = devs
                best_iter = i
                # save best model:
                path = 'best_CNN.pt'
                params = {
                    "state_dict" : model.state_dict(),
                    "optimizer_state" : optimizer.state_dict(),
                    "best_eval" : best_eval,
                    "best_iter" : best_iter
                }
                torch.save(params,path)
                    


                






if __name__ == "__main__":
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = CNN_LSTM_predictor((50,20),3,50)
    model = model.to(device)
    #dfile = '../stock_data/NASDAQ/A'
    #model.prepare_minibatch(dfile)
    #evaluate(model,dfile)
    train(model)
