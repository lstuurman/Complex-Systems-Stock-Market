import pickle
import numpy as np
import glob
import torch
from torch import optim
from torch import nn
import sklearn.preprocessing as sk_prep
import torch.nn.functional as F
from random import shuffle

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
        #inputs = torch.split(batch,1,dim = 1)
        target = targets[b]
        predictions = model(batch)

        for y_i,seq in enumerate(batch):
            seq = seq[:50].cpu().numpy().flatten()
            t = target[y_i].cpu().numpy()[0]
            pred = predictions[y_i].cpu().detach().numpy()[0]
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
    print('Total Correct : ', (corr_bulls + corr_bears)/count)
    return np.average(deviations), corr_bulls,corr_bears,false_bulls,false_bears



def train(model,loss_path,acc_path):
    # devide data : 
    files = glob.glob('../stock_data/NASDAQ/*')
    train = files[:4] # 440
    test = files[5]#[440:450]
    #evl = files[450:]

    # some usefull measures
    training_iters = 10
    train_loss = 0.
    criterion = nn.MSELoss() # loss function
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    best_eval = 1000.
    best_iter = 0
    n_evals = training_iters/50
    losses = []
    eval_data = []

    #scheduler = StepLR(optimizer,step_size = 100,gamma = 0.1)
    los_file = open(loss_path,"a")
    acc_file = open(acc_path,"a")

    for i in range(training_iters):
        for dt,t_file in enumerate(train):
            print('Shuffling training data')
            shuffle(train)
            print(t_file)

            model.train()
            
            minibatches,targets = model.prepare_minibatch(t_file)

            for n,batch in enumerate(minibatches):
                # x = torch.split(batch,1,dim = 1)
                target = targets[n]

                # forward
                outputs = model(batch)

                loss = criterion(outputs,target)
                train_loss += loss.item()
                losses.append(train_loss)

                # backward : 
                model.zero_grad()
                loss.backward()
                optimizer.step()

            # print some info :
            print(dt) 
            if dt % 2 == 0:
                print('Training loss : ',train_loss)
                los_file.write(str(train_loss))
                los_file.write(str('\n'))
                train_loss = 0.

            # evaluate : 
            if dt % 4 == 0: # n_evals == 0:
                print(i)
                devs,cbull,cbear,fbull,fbear = evaluate(model,test) #[int(i/n_evals)])
                ev_data = [devs,cbull,cbear,fbull,fbear]
                eval_data.append([devs,cbull,cbear,fbull,fbear])
                acc_file.write('\t'.join([str(x) for x in ev_data]))
                acc_file.write('\n')

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

    acc_file.close()
    los_file.close()
