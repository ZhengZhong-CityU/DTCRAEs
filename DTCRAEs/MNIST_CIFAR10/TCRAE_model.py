# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('../source/')

from tcn import TemporalConvNet



class TCRAE(nn.Module):
    def __init__(self, device, num_inputs, num_channels, num_hidden, num_classes, kernel_size=2, dropout=0.2, c_dropout=0.0, r_dropout=0.0):
        super(TCRAE, self).__init__()
        self.device=device
        self.num_inputs=num_inputs
        self.num_channels=num_channels
        self.kernel_size=kernel_size
        self.dropout=dropout
        self.c_dropout=c_dropout
        self.r_dropout=r_dropout
        
        self.TCN=TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)   
        self.rnn=nn.GRU(1, num_hidden, batch_first=True)  
        
        # fix the input weights and bias to form a GRU without input
        self.rnn.weight_ih_l0.data.fill_(0)
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.weight_ih_l0.requires_grad=False
        self.rnn.bias_ih_l0.requires_grad=False
        
        self.hidden=nn.Sequential(nn.Linear(num_channels[-1], num_hidden), nn.Tanh())       
        self.c_linear=nn.Linear(num_hidden, num_classes)       
        self.r_linear=nn.Linear(num_hidden, num_inputs)
        
    def forward(self, inputs):
        #inputs shold be the shape of (batch, ninp, L)       
        raw_output=self.TCN(inputs)        
#        raw_output.register_hook(lambda x: print(x[:,:,-1]))
        
        # classification output
        hidden=self.hidden(raw_output[:,:,-1])
        c_output=self.c_linear(F.dropout(hidden, self.c_dropout))
        c_output=F.log_softmax(c_output, dim=1)
        
        # reconstruction output
        x=torch.zeros(inputs.shape[0], inputs.shape[-1], 1).to(self.device)   # Dimension (batch, L, 1)
        rnn_output, h=self.rnn(x, hidden.unsqueeze(0).contiguous()) # rnn_out: Dimension (batch, L, nout)        
        r_output=self.r_linear(F.dropout(rnn_output, self.r_dropout)) # Dimension (batch, L, ninp)         
        r_output=torch.flip(r_output,[1])

        return c_output, r_output.transpose(1,2)  #Dimension (batch, nout) and (batch, ninp, L)
    
if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x=torch.rand(32, 1, 28*28).to(device)
    num_inputs=x.size(1)
    num_channels=[25]*8
    num_hidden=100
    num_classes=10
    kernel_size=6
    model=TCRAE(device, num_inputs, num_channels, num_hidden, num_classes, kernel_size).to(device)
    c_output, r_output=model(x)
    
    print(c_output.shape)
    print(r_output.shape)
    
  