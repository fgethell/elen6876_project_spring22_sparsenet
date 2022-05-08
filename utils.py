import os
import math
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class single_layer(nn.Sequential):
    '''
    This class defines the structure of a single layer in neural network (with or without dropout)
    Structure: -> batch_norm -> ReLU -> Conv2d (3x3) -> dropout ->
    '''
    def __init__(self, num_channels, growth_rate, drop_prob = 0.0):
        super(single_layer, self).__init__()
        
        self.batch_norm1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(num_channels, growth_rate, kernel_size = 3, padding = 1, bias = False)
        
        if drop_prob > 0:
            self.dropout1 = nn.Dropout2d(drop_prob)

            
class bottle_neck(nn.Sequential):
    '''
    This class defines the structure of a single bottleneck layer in neural network (with or without 2 dropouts)
    Structure: -> batch_norm -> ReLU -> Conv2d (1x1) -> dropout -> batch_norm -> ReLU -> Conv2d (3x3) -> dropout ->
    '''
    def __init__(self, num_channels, growth_rate, drop_prob = 0.0):
        
        super(bottle_neck, self).__init__()
        
        internal_channels = 4 * growth_rate
        self.batch_norm1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(num_channels, internal_channels, kernel_size = 1, bias = False)
        
        if drop_prob > 0:
            self.dropout1 = nn.Dropout2d(drop_prob)

        self.batch_norm2 = nn.BatchNorm2d(internal_channels)
        self.relu2 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(internal_channels, growth_rate, kernel_size = 3, padding = 1, bias = False)
        
        if drop_prob > 0:
            self.dropout2 = nn.Dropout2d(drop_prob)          

            
class transition_layer(nn.Sequential):
    '''
    This class defines the structure of a single transition layer in neural network (with or without dropout)
    Structure: -> batch_norm -> ReLU -> Conv2d (1x1) -> dropout -> AvgPool2d (2x2) ->
    '''
    def __init__(self, num_channels, num_output_channels, drop_prob = 0.0):
        
        super(transition_layer, self).__init__()
        
        self.batch_norm1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(num_channels, num_output_channels, kernel_size = 1, bias = False)
        
        if drop_prob > 0.0:
            self.dropout1 = nn.Dropout2d(drop_prob)
            
        self.avgpool = nn.AvgPool2d(2)

        
class flatten_layer(nn.Module):
    '''
    This class defines the structure of a single Flatten layer in neural network (with size equal to input dims)
    '''
    def __init__(self, dim = 0):
        
        super(flatten_layer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.view(x.size(self.dim), -1)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'dim=' + str(self.dim) + ')'
    
def fetch_linear_layer(l):
    return l

def fetch_exponents(l):
    '''
    This function returns the list of exponents of 2 based on layer depth l
    Input: l 
    Output: List containing exponents of 2 till l
    '''
    size = len(l)
    result = []
    
    idx = 1
    while idx <= size:
        result.append(l[size - idx])
        idx = idx * 2
        
    return result


class dense_stage(nn.Sequential):
    '''
    This class initializes the dense portion of the neural network model. Input parameters for the constructor are defined as follows:
    num_dense_blocks: number of dense layers 
    num_channels: number of input channels
    growth_rate: growth rate value
    bottle_neck_flag: flag for bottle neck compression layer 
    drop_prob: 0.0: dropout probability value,
    fetch_type: architecture type (sparse or dense)
    '''
    def __init__(self, num_dense_blocks, num_channels, growth_rate, bottle_neck_flag, drop_prob = 0.0, fetch_type = "sparse"):
        
        super(dense_stage, self).__init__()

        self.num_dense_blocks = num_dense_blocks
        self.previous_channels = [num_channels]
        
        if fetch_type == "dense":
            self.fetch_type = fetch_linear_layer
            
        elif fetch_type == "sparse":
            self.fetch_type = fetch_exponents
        
        else:
            raise NotImplementedError("Wrong fetch type.")

        for i in range(int(num_dense_blocks)):
            
            num_channels = sum(self.fetch_type(self.previous_channels))
            
            if bottle_neck_flag:
                curr_unit = bottle_neck(num_channels, growth_rate, drop_prob)
                
            else:
                curr_unit = single_layer(num_channels, growth_rate, drop_prob)
                
            self.add_module("block-%d" % (i + 1), curr_unit)
            
            self.previous_channels.append(growth_rate)

        self.num_output_channels = sum(self.fetch_type(self.previous_channels))

    def forward(self, x):

        previous_outputs = [x]
        
        for i in range(int(self.num_dense_blocks)):
            
            out = self._modules["block-%d" % (i + 1)](x)
            previous_outputs.append(out)
            
            fetch_outputs = self.fetch_type(previous_outputs)
            
            x = torch.cat(fetch_outputs, 1).contiguous()
            
        return x