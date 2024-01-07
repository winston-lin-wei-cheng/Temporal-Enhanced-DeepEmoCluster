#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wei-Cheng (Winston) Lin
"""
import torch
import torch.nn as nn
import numpy as np
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


__all__ = [ 'VGG', 'vgg16']


def generate_contrast_idx(win_size=2):
    """
      Generate random sample (anchor, positive, negative) batch-set for Triplet-loss:
      e.g., if win_size=2  the definition of {neg, anc, pos} is like:
            seq=[neg neg neg (pos pos anchor pos pos) neg neg] 
    """
    indexes = np.arange(0, 11) # fixed 11 chunks for every utterance
    a_idx = np.random.choice(indexes, 1, replace=False)[0]
    if a_idx==indexes[0]:    # edge case (beginning chunk)
        p_pool = np.arange(a_idx+win_size+1)[1:]
        p_idx = np.random.choice(p_pool, 1, replace=False)[0]
        n_pool = np.array(list(set(indexes)-set(np.append(a_idx, p_pool))))
        n_idx = np.random.choice(n_pool, 1, replace=False)[0]
    elif a_idx==indexes[-1]: # edge case (ending chunk)
        p_pool = np.arange(a_idx, a_idx-win_size-1, step=-1)[1:]
        p_idx = np.random.choice(p_pool, 1, replace=False)[0]
        n_pool = np.array(list(set(indexes)-set(np.append(a_idx, p_pool))))
        n_idx = np.random.choice(n_pool, 1, replace=False)[0]
    else: # middle case
        p_pool = np.arange(a_idx-win_size, a_idx+win_size+1)
        p_pool = np.array([ x for x in p_pool if indexes[0]<=x<=indexes[-1]])
        p_pool = np.delete(p_pool, np.where(p_pool==a_idx))     
        p_idx = np.random.choice(p_pool, 1, replace=False)[0]
        n_pool = np.array(list(set(indexes)-set(np.append(a_idx, p_pool))))
        n_idx = np.random.choice(n_pool, 1, replace=False)[0]
    return a_idx, p_idx, n_idx

class VGG(nn.Module):

    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.cnn_features = features
        self.cnn_out = nn.Sequential(nn.Linear(1024, 256), 
                                     nn.ReLU(inplace=True)) 
        self.classifier = nn.Sequential(nn.Linear(256, 256), 
                                        nn.ReLU(inplace=True))              
        self.emo_regressor = nn.Sequential(nn.Linear(256, 256), 
                                           nn.ReLU(inplace=True))
        self.top_layer_class = nn.Linear(256, num_classes)
        self.top_layer_attri = nn.Linear(256, 1)
        self._initialize_weights()

    def forward(self, x):        
        # shared CNN feature extraction model
        x = self.cnn_features(x)
        x = x.view(x.size(0), -1)
        x = self.cnn_out(x)
        # consider temporal info between chunks (using Triplet approach)        
        x = x.view(-1, 11, x.size(1))
        anchor = torch.empty(size=(0, x.size(2)), requires_grad=True).to(device)
        positive = torch.empty(size=(0, x.size(2)), requires_grad=True).to(device)
        negative = torch.empty(size=(0, x.size(2)), requires_grad=True).to(device)
        for re in range(10): # each sentence random samples 10 Triplet pairs
            a_idx, p_idx, n_idx = generate_contrast_idx(win_size=2)
            anchor = torch.cat((anchor, x[:, a_idx, :]),dim=0)
            positive = torch.cat((positive, x[:, p_idx, :]),dim=0)
            negative = torch.cat((negative, x[:, n_idx, :]),dim=0)                    
        x = x.reshape(-1, x.size(2))
        # for deep-cluster classification
        x_class = self.classifier(x)
        if self.top_layer_class:
            x_class = self.top_layer_class(x_class)
        # for emotion regression
        x_attri = self.emo_regressor(x)  
        x_attri = self.top_layer_attri(x_attri) 
        return anchor, positive, negative, x_class, x_attri

    def _initialize_weights(self):
        for y,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(input_dim, batch_norm):
    layers = []
    in_channels = input_dim
    cfg = [32, 32, '64', '64', '128', '128', '128']
    for v in cfg:
        # max pooling layer
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            
        # conv2D with stride 2
        elif (type(v) == str)&(v!='M'):
            conv2d = nn.Conv2d(in_channels, int(v), kernel_size=3, stride=2, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(int(v)), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = int(v)                  
        
        # conv2D with stride 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg16(bn, out):
    inp_dim = 1
    model = VGG(make_layers(inp_dim, bn), out)
    return model
