'''
Defines the DanQ model architecture
'''
import sys
import numpy as np
import h5py
import torch
import torch.nn as nn

class DanQ(torch.nn.Module):
    def __init__(self):
        super(DanQ,self).__init__()

        #Conv layer
        self.conv_layer=nn.Conv1d(in_channels=4,out_channels=320,kernel_size=26,padding=0)
        self.relu=nn.ReLU
        
        #max pool
        self.pool=nn.MaxPool1d(kernel_size=13, stride=13)
        self.dropout=nn.Dropout(0.2)

        
        #BiLSTM
        self.biLSTM=nn.LSTM(input_size=320,hidden_size=320,num_layers=1,bidirectional=True,batch_first=True)
        self.dropout2=nn.Dropout(0.2)

        #fully connected dense
        self.dense=nn.Linear(75*640, 925)
        self.dense_activation=nn.ReLU

        #output
        self.output=nn.Linear(925,39)
        self.output_activation=nn.Sigmoid


    def forward(self,x):

        #conv
        x=self.conv_layer(x)
        x=self.relu

        #max pool
        x=self.pool(x)
        x=self.dropout(x)

        #biLSTM
        #reshape first 
        x=x.permute(0,2,1)
        x,_=self.biLSTM(x)
        x=self.dropout2(x)

        #dense
        #flatten first
        x=x.reshape(x.size(0),-1)

        #dense
        x=self.dense(x)
        x=self.dense_activation(x)

        #output
        x=self.output(x)
        x=self.output_activation(x)

