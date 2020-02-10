from helpers_LSTMsent import *
from Sent_LSTM import *
import time
import re
import random
import time
import math
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set_style('darkgrid')
import pickle
import pandas

import torch
from torch import optim
from torch import nn

### TRAINING ####

def train_model(model, optimizer, num_iterations=30000, 
                print_every=1000, eval_every=1000,
                batch_fn=get_minibatch, 
                prep_fn=prepare_minibatch,
                eval_fn=evaluate,
                batch_size=25, eval_batch_size=None):
  """Train a model. /home/lau/GIT/Complex Systems Stock Market/tweet_training_data/trees"""  
  train_data = list(examplereader("../tweet_training_data/trees/train.txt",lower=False))
  dev_data = list(examplereader("../tweet_training_data/trees/dev.txt", lower=False))
  test_data = list(examplereader("../tweet_training_data/trees/test.txt", lower=False))
  iter_i = 0
  train_loss = 0.
  print_num = 0
  start = time.time()
  criterion = nn.CrossEntropyLoss() # loss function
  best_eval = 0.
  best_iter = 0
  
  # store train loss and validation accuracy during training
  # so we can plot them afterwards
  losses = []
  accuracies = []  
  
  if eval_batch_size is None:
    eval_batch_size = batch_size
  
  while True:  # when we run out of examples, shuffle and continue
    for batch in batch_fn(train_data, batch_size=batch_size):

      # forward pass
      model.train()
      x, targets = prep_fn(batch, model.vocab)
      #print(torch.cuda.get_device_name(torch.cuda.current_device()))
      
      logits = model(x)

      B = targets.size(0)  # later we will use B examples per update
      
      # compute cross-entropy loss (our criterion)
      # note that the cross entropy loss function computes the softmax for us
      loss = criterion(logits.view([B, -1]), targets.view(-1))
      train_loss += loss.item()

      # backward pass
      # erase previous gradients
      model.zero_grad()
      
      # compute gradients
      loss.backward()

      # update weights - take a small step in the opposite dir of the gradient
      optimizer.step()

      print_num += 1
      iter_i += 1

      # print info
      if iter_i % print_every == 0:
        print("Iter %r: loss=%.4f, time=%.2fs" % 
              (iter_i, train_loss, time.time()-start))
        losses.append(train_loss)
        print_num = 0        
        train_loss = 0.

      # evaluate
      if iter_i % eval_every == 0:
        _, _, accuracy = eval_fn(model, dev_data, batch_size=eval_batch_size,
                                 batch_fn=batch_fn, prep_fn=prep_fn)
        accuracies.append(accuracy)
        print("iter %r: dev acc=%.4f" % (iter_i, accuracy))       
        
        # save best model parameters
        if accuracy > best_eval:
          print("new highscore")
          best_eval = accuracy
          best_iter = iter_i
          path = "{}.pt".format(model.__class__.__name__)
          ckpt = {
              "state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "best_eval": best_eval,
              "best_iter": best_iter
          }
          torch.save(ckpt, path)

      # done training
      if iter_i == num_iterations:
        print("Done training")
        
        # evaluate on train, dev, and test with best model
        print("Loading best model")
        path = "{}.pt".format(model.__class__.__name__)        
        ckpt = torch.load(path)
        model.load_state_dict(ckpt["state_dict"])
        
        _, _, train_acc = eval_fn(
            model, train_data, batch_size=eval_batch_size, 
            batch_fn=batch_fn, prep_fn=prep_fn)
        _, _, dev_acc = eval_fn(
            model, dev_data, batch_size=eval_batch_size,
            batch_fn=batch_fn, prep_fn=prep_fn)
        _, _, test_acc = eval_fn(
            model, test_data, batch_size=eval_batch_size, 
            batch_fn=batch_fn, prep_fn=prep_fn)
        
        print("best model iter {:d}: "
              "train acc={:.4f}, dev acc={:.4f}, test acc={:.4f}".format(
                  best_iter, train_acc, dev_acc, test_acc))
        
        return losses, accuracies, model

if __name__ == "__main__":
    # init model: 
    lstm = LSTMClassifier(300,168,5)
    lstm.create_vocab()
    vectors = lstm.vectors
    # try cuda ...
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #'cuda' if torch.cuda.is_available() else 

    # copy embeddings into model:
    with torch.no_grad():
        lstm.embed.weight.data.copy_(torch.from_numpy(vectors))
        lstm.embed.weight.requires_grad = False

    lstm = lstm.to(device)
    batch_size = 25
    optimizer = optim.Adam(lstm.parameters(), lr = 2e-4)

    # train :::
    losses,accuracies,best = train_model(lstm,optimizer,
        num_iterations=20000,print_every=500,
        eval_every=500, batch_size=batch_size)
    
    # save model:
    pickle.dump(best,open('best_sentiment_LSTM.pkl','wb'))
    # save dataframe of training : 
    df = pandas.DataFrame(list(zip(accuracies,losses)),columns = ['Accuracy','Loss'])
    df.to_csv('training_data.csv')

