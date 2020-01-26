# Class to for pytorch sentiment LSTM NN
import pandas as pd
import re
import random
import time
import math
import numpy as np
import nltk

import torch
from torch import optim
from torch import nn

class LSTMClassifier(nn.Module):
  """Encodes sentence with an LSTM and projects final hidden state"""

  def __init__(self, embedding_dim, hidden_dim, output_dim):
    super(LSTMClassifier, self).__init__()
    # vocabulary : 
    self.vocab = None
    self.vocab_size = None

    # NN : 
    self.hidden_dim = hidden_dim
    self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
    self.rnn = nn.LSTM(embedding_dim,hidden_dim)
    #self.rnn = MyLSTMCell(embedding_dim, hidden_dim)
    
    self.output_layer = nn.Sequential(     
        nn.Dropout(p=0.5),  # explained later
        nn.Linear(hidden_dim, output_dim)
    )

  def forward(self, x):
    
    B = x.size(0)  # batch size (this is 1 for now, i.e. 1 single example)
    T = x.size(1)  # time (the number of words in the sentence)
    
    input_ = self.embed(x)

    # here we create initial hidden states containing zeros
    # we use a trick here so that, if input is on the GPU, then so are hx and cx
    hx = input_.new_zeros(B, self.rnn.hidden_size)
    cx = input_.new_zeros(B, self.rnn.hidden_size)
    
    # process input sentences one word/timestep at a time
    # input is batch-major, so the first word(s) is/are input_[:, 0]
    outputs = []   
    for i in range(T):
      hx, cx = self.rnn(input_[:, i], (hx, cx))
      outputs.append(hx)
    
    # if we have a single example, our final LSTM state is the last hx
    if B == 1:
      final = hx
    else:
      
      # two lines below not needed if using LSTM form pytorch
      outputs = torch.stack(outputs, dim=0)          # [T, B, D]
      outputs = outputs.transpose(0, 1).contiguous()  # [B, T, D]
      
      # to be super-sure we're not accidentally indexing the wrong state
      # we zero out positions that are invalid
      pad_positions = (x == 1).unsqueeze(-1)
      
      outputs = outputs.contiguous()      
      outputs = outputs.masked_fill_(pad_positions, 0.)
        
      mask = (x != 1)  # true for valid positions [B, T]
      lengths = mask.sum(dim=1)                  # [B, 1]

      indexes = (lengths - 1) + torch.arange(B, device=x.device, dtype=x.dtype) * T
      final = outputs.view(-1, self.hidden_dim)[indexes]  # [B, D]
    
    # we use the last hidden state to classify the sentence
    logits = self.output_layer(final)
    return logits

    def create_vocab():
        # read trianing data : 
        f_name = '/home/lau/GIT/Complex Systems Stock Market/tweet_training data'
        test_df = pd.read_csv(f_name)
        df.head()

if __name__ == "__main__":
    #(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab)
    vocab_size = 18000
    output_dim = 2
    model = LSTMClassifier( 300, 168, output_dim)
    model.create_vocab()