import torch
from torch import nn,optim
import math

class MyLSTMCell(nn.Module):
  """Our own LSTM cell"""

  def __init__(self, input_size, hidden_size, bias=True):
    """Creates the weights for this LSTM"""
    super(MyLSTMCell, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    # Weights for input : 
    self.Wi = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
    self.Bi = nn.Parameter(torch.Tensor(4 * hidden_size))

    # Weights for preveious cell state ;
    self.Wh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
    self.Bh = nn.Parameter(torch.Tensor(4 * hidden_size))

    self.reset_parameters()

  def reset_parameters(self):
    """This is PyTorch's default initialization method"""
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)  

  def forward(self, input_, hx, mask=None):
    """
    input is (batch, input_size)
    hx is ((batch, hidden_size), (batch, hidden_size))
    """
    prev_h, prev_c = hx

    x = input_

    # two linear transformations
    linear_transform = (x @ self.Wi + self.Bi) + (prev_h @ self.Wh + self.Bh)
    # chunking into corresponding gates : 
    i,f,g,o = linear_transform.chunk(4, 1)
    # activation (sigmoid or tanh)
    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)
    # calculation of new cell states : 
    c = f * prev_c + i * g 
    h = o * torch.tanh(c)

    return h, c
  


  def __repr__(self):
    return "{}({:d}, {:d})".format(
        self.__class__.__name__, self.input_size, self.hidden_size)