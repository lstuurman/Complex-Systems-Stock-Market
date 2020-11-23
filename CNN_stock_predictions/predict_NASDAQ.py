import torch
from torch import optim
from torch import nn
import sklearn.preprocessing as sk_prep
import numpy as np
import pandas as pd
import glob
import datetime as dt  
from simple_CNN2 import simple_CNN

def load_latest_stocks():
    files = glob.glob('../stock_data/NASDAQ_DAILY/*')
    # return tuples of (tick,[[price,data]]) of last 50 time points
    data_list = []
    for f in files:
        data = pd.read_csv(f,index_col = 0)[-50:]
        #check if data up to date : 
        today = (dt.datetime.today()- dt.timedelta(days=1)).strftime("%Y-%m-%d")
        if str(data.index[-1]) != today:
            pass

        price,volume = data.to_numpy().T
        #print(price.shape,volume.shape)

        vol_slice = sk_prep.minmax_scale(volume.reshape(-1,1),copy = True)
        pr_slice = sk_prep.minmax_scale(price.reshape(-1,1),copy = True)
        concat = np.concatenate((pr_slice,vol_slice),axis = 0)
        data_list.append((f.split('/')[-1],concat))
        #print((f.split('/')[-1],concat))
        #print(concat.shape)
    return data_list


def predict_stocks():
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    # load desired model : 
    model = simple_CNN((50,50),30)
    model_path = 'best__simple_CNN2.pt'
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()

    # load data : 
    latest_data = load_latest_stocks()

    predictions = []
    for dtpoint in latest_data:
        x = torch.Tensor(dtpoint[1]).to(device)
        prdct = model(x).detach().numpy()
        last_price = dtpoint[1][50]
        diff = last_price - prdct
        #print(last_price,prdct,diff)
        predictions.append((dtpoint[0],diff[0,0]))

    predictions.sort(key = lambda x: x[1])
    to_save = predictions[:10] + predictions[-10:]
    fname = 'PREDICTIONS ' + dt.datetime.today().strftime("%Y-%m-%d")
    with open(fname,'w') as f:
        for x in to_save:
            f.writelines(str(x[0]) + '\t' + str(x[1]) + '\n')
        #f.write([str(x) + '\n' for x in to_save])


predict_stocks()
#load_latest_stocks()