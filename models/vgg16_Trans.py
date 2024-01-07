#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wei-Cheng (Winston) Lin
"""
import torch
import torch.nn as nn
import math
from torch import Tensor


__all__ = [ 'VGG', 'vgg16']

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 11):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class VGG(nn.Module):

    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.cnn_features = features
        self.cnn_out = nn.Sequential(nn.Linear(1024, 256), 
                                     nn.ReLU(inplace=True), 
                                     nn.Dropout(0.5))
        self.pos_enc = PositionalEncoding(d_model=256, dropout=0)
        # Transformer-based temporal network
        self.temporal_net = nn.TransformerEncoderLayer(d_model=256, nhead=1, dim_feedforward=256, dropout=0.3)
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
        # consider temporal info between chunks (using Transformer-based model)     
        # NOTE: Pytorch 1.4.0 version doesn't support batch_first=True, so we use permutation  
        # to obtain [seq_len, batch_size, feat_dim] shape
        x = x.view(-1, 11, x.size(1))
        x = x.permute(1, 0, 2)
        x = self.pos_enc(x)  
        x = self.temporal_net(x)
        x = x.permute(1, 0, 2)            # resume [batch_size, seq_len, feat_dim] shape
        x = x.reshape(-1, x.size(2))      # reshape to [batch_size*seq_len, feat_dim]   
        # for deep-cluster classification
        x_class = self.classifier(x)
        if self.top_layer_class:
            x_class = self.top_layer_class(x_class)
        # for emotion regression
        x_attri = self.emo_regressor(x)  
        x_attri = self.top_layer_attri(x_attri)  
        return x_class, x_attri

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
